# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any, List, Tuple

import re
import concurrent.futures
import ast
import os
import time

# Add OpenAI import for vLLM process evaluation
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("Warning: OpenAI package not found. Process evaluation will be disabled.")

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _extract_tool_pairs(text: str) -> List[str]:
    """Extract <tool_call>...</tool_call> and <tool_response>...</tool_response> pairs."""
    if not isinstance(text, str):
        return []
    pattern = re.compile(r"<tool_call>.*?</tool_call>.*?<tool_response>.*?</tool_response>", re.DOTALL | re.IGNORECASE)
    return [m.group(0).strip() for m in pattern.finditer(text or "")]


def _count_tool_use(text: str) -> int:
    """Count the number of tool uses in the text."""
    if not isinstance(text, str):
        return 0
    # Count <tool_call> tags (similar to count_num_tool_use in code.py)
    pattern = r"<tool_call>"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return len(matches)


# System prompt for LLM-as-judge (from simple_vllm_chat.py)
JUDGE_SYSTEM_PROMPT = """
You are an evaluation assistant. Your goal is to assess how well a large language model, using search tools, answered a user's factual question.

Evaluate each <tool_call>…</tool_call> and its following <tool_response>…</tool_response> pair. For EACH pair, score the following three dimensions on a 0–2 scale (integers):

- factual_correctness:
  0: incorrect or misleading
  1: partially correct or incomplete
  2: fully correct and well-supported

- search_efficiency:
  0: ineffective or irrelevant search
  1: somewhat effective but noisy or redundant
  2: highly effective and focused

- answer_clarity:
  0: confusing or fails to answer
  1: understandable but needs clarity or structure
  2: clear, well-organized, concise

Output requirements:
First provide reasoning per pair as structured chain-of-thought, citing evidence. Then output ratings in the exact template:

<think>
Pair 1 reasoning…
Pair 2 reasoning…
…
</think>
<answer>
ratings_by_pair=[[r1_cor,r1_eff,r1_clar],[r2_cor,r2_eff,r2_clar],...]
</answer>

"""


def _assemble_judge_user_prompt(question: str, tool_pairs: List[str]) -> str:
    """Assemble user prompt for LLM-as-judge following simple_vllm_chat.py format."""
    base_prompt = f"Question:\n{question}\n\nSearch engine execution part:"
    if not tool_pairs:
        return base_prompt
    return base_prompt + "\n" + "\n\n".join(tool_pairs)


def _extract_process_scores(content: str) -> List[Tuple[float, float, float]]:
    """
    Extract and parse process evaluation scores from LLM response (from simple_vllm_chat.py).
    
    Args:
        content: Raw response content from LLM
        
    Returns:
        List of triplets (factual_correctness, search_efficiency, answer_clarity) normalized to [0, 1]
    """
    # First try to extract from <answer> block as expected
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    
    if answer_match:
        search_text = answer_match.group(1).strip()
    else:
        search_text = content
    
    # Try improved regex patterns
    patterns_to_try = [
        r"ratings_by_pair\s*=\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])",  # Nested bracket matching
        r"ratings_by_pair\s*=\s*(\[\s*\[.*?\]\s*(?:\s*,\s*\[.*?\]\s*)*\])",  # Simple nested pattern
        r"ratings_by_pair\s*=\s*(\[.*?\]\s*(?:\s*,\s*\[.*?\]\s*)*)",  # Fallback without outer brackets
    ]
    
    raw_list = None
    for pattern in patterns_to_try:
        match = re.search(pattern, search_text, re.DOTALL)
        if match:
            ratings_str = match.group(1)
            try:
                raw_list = ast.literal_eval(ratings_str)
                break
            except Exception:
                try:
                    raw_list = eval(ratings_str)
                    break
                except Exception:
                    continue
    
    # Fallback: Manual bracket counting
    if raw_list is None:
        start_match = re.search(r"ratings_by_pair\s*=", search_text)
        if not start_match:
            return []
        
        start_pos = start_match.end()
        text_from_start = search_text[start_pos:].strip()
        
        if not text_from_start.startswith('['):
            return []
        
        bracket_count = 0
        end_pos = 0
        
        for i, char in enumerate(text_from_start):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_pos = i + 1
                    break
        
        if bracket_count != 0:
            return []
        
        ratings_str = text_from_start[:end_pos]
        try:
            raw_list = ast.literal_eval(ratings_str)
        except Exception:
            try:
                raw_list = eval(ratings_str)
            except Exception:
                return []
    
    if not isinstance(raw_list, list):
        return []
    
    # Normalize to [0,1] and coerce to triplets
    normalized: List[Tuple[float, float, float]] = []
    
    for item in raw_list:
        try:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            
            # Normalize from 0-2 scale to 0-1 scale
            fc, se, ac = float(item[0]) / 2.0, float(item[1]) / 2.0, float(item[2]) / 2.0
            normalized.append((fc, se, ac))
        except Exception:
            continue
    
    return normalized


def _save_sample_data(data, reward_tensor, main_extra_by_index, auxiliary_rewards_by_index, seg_ids_by_index, tokenizer, max_samples: int = 3):
    """
    Save detailed sample data for debugging and analysis (similar to code.py).
    Only saves samples with multiple segments or interesting patterns.
    """
    try:
        # Create output directory if not exists
        output_dir = "reward_samples_naive_llm"
        os.makedirs(output_dir, exist_ok=True)
        
        saved_count = 0
        timestamp = int(time.time())
        
        # Iterate through data to find interesting samples
        for i in range(min(len(data), 20)):  # Check first 20 samples
            if saved_count >= max_samples:
                break
                
            data_item = data[i]
            
            # Check if this sample has seg_ids and multiple tool use segments
            seg_ids = seg_ids_by_index.get(i, [])
            if not seg_ids:
                continue
                
            unique_segments = set(seg_id for seg_id in seg_ids if seg_id >= 0)  # Ignore -1 padding
            
            # Only save samples with 2 or more tool use segments (multiple tool calls)
            if len(unique_segments) >= 2:
                filename = f"{output_dir}/naive_llm_sample_{saved_count}_{timestamp}.txt"
                
                # Get response data
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum().item()
                valid_response_ids = response_ids[:valid_response_length]
                
                # Get response_mask from batch (computed by trainer)
                # Use the response_mask that was already computed by trainer
                if "response_mask" in data_item.batch:
                    response_mask_tensor = data_item.batch["response_mask"][:valid_response_length]
                    response_mask = response_mask_tensor.cpu().tolist()
                else:
                    # Fallback: compute from attention_mask if response_mask not available
                    full_attention_mask = data_item.batch["attention_mask"]
                    response_attention_mask = full_attention_mask[prompt_length:prompt_length + valid_response_length]
                    response_mask = response_attention_mask.cpu().tolist()
                
                # Decode tokens
                decoded_response = tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                
                # Get individual token strings
                token_strings = []
                for token_id in valid_response_ids:
                    token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                    token_strings.append(token_str)
                
                # Get rewards (sequence rewards + final reward)
                sequence_rewards = reward_tensor[i, :valid_response_length].cpu().tolist()
                final_reward = reward_tensor[i, -1].item()
                
                # Get additional info
                test_case = data_item.non_tensor_batch.get("test_cases", {})
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
                data_source = data_item.non_tensor_batch.get("data_source", "")
                
                # Get main extra info for this sample
                main_extra = main_extra_by_index.get(i, {})
                
                # Check auxiliary rewards for this sample
                auxiliary_rewards = auxiliary_rewards_by_index.get(i, [])
                has_auxiliary = any(r != 0.0 for r in auxiliary_rewards)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=== NAIVE LLM REWARD MANAGER SAMPLE DATA ===\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Sample Index: {i}\n")
                    f.write(f"Unique tool use segments: {sorted(unique_segments)} (Total: {len(unique_segments)})\n")
                    f.write(f"Has auxiliary rewards: {has_auxiliary}\n\n")
                    
                    f.write("=== DECODED RESPONSE ===\n")
                    f.write(f"Length: {len(valid_response_ids)} tokens\n")
                    f.write(f"Decoded: {decoded_response}\n\n")
                    
                    f.write("=== TOKEN-LEVEL DETAILS ===\n")
                    f.write("Format: idx | token_id | token_string | seg_id | reward | aux_reward | response_mask\n")
                    f.write("-" * 100 + "\n")
                    
                    for j, (token_id, token_str, reward) in enumerate(zip(valid_response_ids, token_strings, sequence_rewards)):
                        seg_id_str = str(seg_ids[j]) if j < len(seg_ids) else "N/A"
                        aux_reward = auxiliary_rewards[j] if j < len(auxiliary_rewards) else 0.0
                        response_mask_val = response_mask[j] if j < len(response_mask) else "N/A"
                        
                        f.write(f"{j:3d}: {token_id:6d} | {repr(token_str):20s} | {seg_id_str:>5s} | {reward:8.4f} | {aux_reward:8.4f} | {response_mask_val:>12}\n")
                    
                    f.write(f"\n=== REWARDS SUMMARY ===\n")
                    non_zero_rewards = [(j, r) for j, r in enumerate(sequence_rewards) if r != 0.0]
                    f.write(f"Non-zero sequence reward positions: {len(non_zero_rewards)}\n")
                    for pos, reward in non_zero_rewards:
                        f.write(f"  Position {pos}: {reward:.4f}\n")
                    
                    f.write(f"Final reward (T+1): {final_reward:.4f}\n")
                    
                    non_zero_aux = [(j, r) for j, r in enumerate(auxiliary_rewards) if r != 0.0]
                    f.write(f"Non-zero auxiliary positions: {len(non_zero_aux)}\n")
                    for pos, reward in non_zero_aux:
                        f.write(f"  Position {pos}: {reward:.4f}\n")
                    
                    f.write(f"\n=== SEGMENT ANALYSIS ===\n")
                    f.write(f"Segment IDs: {seg_ids}\n")
                    
                    # Count tokens per segment
                    segment_counts = {}
                    for seg_id in seg_ids:
                        if seg_id >= 0:  # Ignore -1 padding
                            segment_counts[seg_id] = segment_counts.get(seg_id, 0) + 1
                    
                    f.write(f"Segment distribution:\n")
                    for seg_id in sorted(segment_counts.keys()):
                        f.write(f"  Segment {seg_id}: {segment_counts[seg_id]} tokens\n")
                    
                    # Find segment boundaries
                    boundaries = []
                    if seg_ids:
                        current_seg = seg_ids[0]
                        for j in range(1, len(seg_ids)):
                            if seg_ids[j] != current_seg and seg_ids[j] >= 0:
                                boundaries.append(j - 1)
                                current_seg = seg_ids[j]
                        boundaries.append(len(seg_ids) - 1)
                    
                    f.write(f"Segment boundaries (end positions): {boundaries}\n")
                    
                    f.write(f"\n=== METADATA ===\n")
                    f.write(f"Data source: {data_source}\n")
                    f.write(f"Ground truth: {ground_truth}\n")
                    f.write(f"Test case: {test_case}\n")
                    f.write(f"Extra info: {extra_info}\n")
                    
                    f.write(f"\n=== MAIN SCORE EXTRA INFO ===\n")
                    if isinstance(main_extra, dict) and main_extra:
                        for key, value in main_extra.items():
                            f.write(f"{key}: {value}\n")
                    else:
                        f.write("No extra info from main score\n")
                    
                    # Summary statistics
                    total_reward = sum(sequence_rewards)
                    total_aux_reward = sum(auxiliary_rewards)
                    
                    f.write(f"\n=== SUMMARY ===\n")
                    f.write(f"Total sequence reward: {total_reward:.4f}\n")
                    f.write(f"Total auxiliary reward: {total_aux_reward:.4f}\n")
                    f.write(f"Final reward (T+1): {final_reward:.4f}\n")
                    f.write(f"Combined total: {total_reward + final_reward:.4f}\n")
                    f.write(f"Total tokens: {len(valid_response_ids)}\n")
                    f.write(f"Total segments: {len(unique_segments)}\n")
                    
                print(f"Sample data saved to: {filename}")
                saved_count += 1
        
        if saved_count == 0:
            print("No samples with multiple tool use segments found, no data saved.")
        else:
            print(f"Saved {saved_count} sample(s) with multiple tool use segments to {output_dir}/")
            
    except Exception as e:
        print(f"Failed to save sample data: {e}")
        import traceback
        traceback.print_exc()


def _compute_execution_scores_from_pairs(text: str) -> List[float]:
    """
    Heuristic execution scores per <tool_call>/<tool_response> pair.
    0.5 for seemingly OK results, 0.0 if any negative signal is detected inside a pair.
    """
    pairs = _extract_tool_pairs(text or "")
    if not pairs:
        # Fallback to <execute>/<results> tags if present in other formats
        pattern = re.compile(r"<execute>.*?</execute>.*?<results>.*?</results>", re.DOTALL | re.IGNORECASE)
        pairs = [m.group(0).strip() for m in pattern.finditer(text or "")]

    scores: List[float] = []
    if not pairs:
        return scores

    negative_keywords = [
        r"error", r"exception", r"traceback",
        r"timeout", r"time limit", r"time-limit", r"killed", r"memoryerror", r"memory limit",
        r"syntaxerror", r"indentationerror", r"invalid syntax", r"compilation_error", r"runtime_error",
        r"worker terminated", r"process pool", r"terminated abruptly", r"pool crashed", r"operation blocked", r"restricted",
        r"unknown_error", r"output_mismatch",
    ]
    negative_re = re.compile(r"(" + r"|".join(negative_keywords) + r")", re.IGNORECASE)

    results_pattern = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL | re.IGNORECASE)
    results_alt_pattern = re.compile(r"<results>(.*?)</results>", re.DOTALL | re.IGNORECASE)

    for pair in pairs:
        m = results_pattern.search(pair) or results_alt_pattern.search(pair)
        if not m:
            scores.append(0.0)
            continue
        cleaned_text = (m.group(1) or "").strip()
        if not cleaned_text:
            scores.append(0.0)
            continue
        if negative_re.search(cleaned_text):
            scores.append(0.0)
        else:
            scores.append(0.5)

    return scores


def _build_seg_ids_from_tokens(token_strings: List[str]) -> List[int]:
    """
    Build seg_ids for each token. Each segment runs until a closing tool response tag
    ("</tool_response>" or "</results>"). The final tail segment uses -1.
    If no tool responses are found, all tokens are -1.
    """
    resp_len = len(token_strings)
    if resp_len == 0:
        return []

    closing_tags = ["</tool_response>", "</results>"]
    boundaries: List[int] = []  # inclusive end index of each non-final segment

    # Rolling buffer limited to the maximum closing tag length
    # Add some extra buffer to handle cross-token boundaries
    max_tag_len = max(len(tag) for tag in closing_tags)
    buffer_size = max_tag_len + 20  # Extra buffer for safety
    buffer = ""
    for idx, tok in enumerate(token_strings):
        buffer += tok
        # Keep only the last buffer_size characters
        if len(buffer) > buffer_size:
            buffer = buffer[-buffer_size:]
        for tag in closing_tags:
            if buffer.endswith(tag):
                boundaries.append(idx)
                break

    seg_ids = [-1] * resp_len
    if not boundaries:
        return seg_ids

    start = 0
    for seg_index, end_idx in enumerate(boundaries):
        end_idx = min(end_idx, resp_len - 1)
        for pos in range(start, end_idx + 1):
            seg_ids[pos] = seg_index
        start = end_idx + 1

    # Remaining tokens stay -1 (final segment)
    return seg_ids


def _compute_segment_boundaries(seg_ids: List[int]) -> List[int]:
    """Return end indices for contiguous runs with seg_id >= 0."""
    if not seg_ids:
        return []
    resp_len = len(seg_ids)
    boundaries: List[int] = []
    in_valid = seg_ids[0] >= 0
    for j in range(1, resp_len):
        if seg_ids[j] != seg_ids[j - 1]:
            if in_valid:
                boundaries.append(j - 1)
            in_valid = seg_ids[j] >= 0
    if in_valid:
        boundaries.append(resp_len - 1)
    return boundaries


def _batch_llm_judge_evaluate(batch_requests: List[dict], vllm_client, vllm_model: str, judge_timeout: float, judge_max_retries: int) -> List[dict]:
    """
    Batch evaluate multiple samples using LLM-as-judge.
    
    Args:
        batch_requests: List of dicts with keys: 'question', 'tool_pairs', 'index'
        vllm_client: Shared vLLM client
        vllm_model: Model name
        judge_timeout: Timeout for each request
        judge_max_retries: Max retry attempts
    
    Returns:
        List of dicts with keys: 'index', 'score_triplets', 'success'
    """
    if not batch_requests or not vllm_client:
        return []
    
    # Prepare batch messages
    batch_messages = []
    request_map = {}  # Map batch index to original request
    
    for batch_idx, request in enumerate(batch_requests):
        question = request['question']
        tool_pairs = request['tool_pairs']
        
        if tool_pairs:  # Only process requests with tool pairs
            user_prompt = _assemble_judge_user_prompt(question, tool_pairs)
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            batch_messages.append(messages)
            request_map[len(batch_messages) - 1] = request
    
    results = []
    if not batch_messages:
        return results
    
    print(f"Batch evaluating {len(batch_messages)} samples with LLM-judge...")
    
    # Process batch (most OpenAI-compatible APIs don't support true batch completion)
    # So we use concurrent individual requests instead
    import concurrent.futures
    
    def evaluate_single(messages_with_idx):
        msg_idx, messages = messages_with_idx
        request = request_map[msg_idx]
        
        for attempt in range(judge_max_retries):
            try:
                completion = vllm_client.chat.completions.create(
                    model="qwen2.5-7b-instruct",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2048,
                    timeout=judge_timeout
                )
                
                content = completion.choices[0].message.content
                if content:
                    score_triplets = _extract_process_scores(content)
                    return {
                        'index': request['index'],
                        'score_triplets': score_triplets,
                        'success': True
                    }
                else:
                    continue  # Try again if no content
                    
            except Exception as e:
                if attempt == judge_max_retries - 1:
                    print(f"LLM-judge failed for index {request['index']} after {judge_max_retries} attempts: {e}")
                    return {
                        'index': request['index'],
                        'score_triplets': [],
                        'success': False
                    }
                else:
                    time.sleep(0.5)  # Brief delay before retry
        
        return {
            'index': request['index'],
            'score_triplets': [],
            'success': False
        }
    
    # Use a smaller thread pool for LLM requests to avoid overwhelming the server
    max_concurrent = min(8, len(batch_messages))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        indexed_messages = list(enumerate(batch_messages))
        futures = [executor.submit(evaluate_single, msg_with_idx) for msg_with_idx in indexed_messages]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Batch LLM-judge evaluation error: {e}")
    
    return results


def _per_token_execution_rewards(seg_ids: List[int], execution_scores: List[float]) -> List[float]:
    """
    Create a per-token reward vector by assigning execution_scores to each non-negative
    segment and 0 to the final -1 segment. Extra scores are accumulated to the last segment.
    """
    resp_len = len(seg_ids)
    rewards = [0.0] * resp_len

    # Build non -1 segment boundaries
    boundaries = _compute_segment_boundaries(seg_ids)
    if not boundaries:
        return rewards

    num_segments = len(boundaries)
    per_segment_execution = [0.0] * num_segments
    if execution_scores:
        k = min(len(execution_scores), num_segments)
        for i in range(k):
            per_segment_execution[i] = float(execution_scores[i])
        if len(execution_scores) > num_segments:
            per_segment_execution[-1] += sum(float(v) for v in execution_scores[num_segments:])

    # Fill rewards for each segment
    start_idx = 0
    for seg_idx, end_idx in enumerate(boundaries):
        seg_reward = per_segment_execution[seg_idx]
        for pos in range(start_idx, end_idx + 1):
            rewards[pos] += seg_reward
        start_idx = end_idx + 1

    return rewards




def _process_chunk_worker(chunk_tasks: List[Tuple[int, List[str], str, str, Any, str, dict, Any, bool, bool, str, str, float, int]]):
    """Process a chunk of tasks in a single process using separate thread pools for auxiliary and main tasks.
    
    Reuses vLLM client per process and implements timeout/retry for LLM-judge calls.
    Returns 5 dicts keyed by index to reduce cross-process object count.
    """
    seg_ids_by_index: dict[int, List[int]] = {}
    auxiliary_rewards_by_index: dict[int, List[float]] = {}
    main_score_by_index: dict[int, float] = {}
    main_extra_by_index: dict[int, dict] = {}
    stats_by_index: dict[int, dict] = {}

    if not chunk_tasks:
        return seg_ids_by_index, auxiliary_rewards_by_index, main_score_by_index, main_extra_by_index, stats_by_index

    # Initialize per-process vLLM client for reuse
    vllm_client = None
    enable_llm_judge = any(task[10] for task in chunk_tasks)  # task[10] is enable_llm_judge
    if enable_llm_judge and OpenAI is not None:
        vllm_api_base = chunk_tasks[0][11] if chunk_tasks else None  # task[11] is vllm_api_base
        if vllm_api_base:
            try:
                import httpx
                # Optimize connection pool for concurrent requests
                http_client = httpx.Client(
                    limits=httpx.Limits(
                        max_keepalive_connections=20,
                        max_connections=100,
                        keepalive_expiry=30.0
                    ),
                    timeout=httpx.Timeout(60.0),
                )
                vllm_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
                    base_url="http://10.24.1.204:9000/v1",
                    http_client=http_client,
                )
                print(f"Initialized shared vLLM client for {len(chunk_tasks)} tasks in process")
                
                # Register cleanup function
                import atexit
                atexit.register(lambda: http_client.close() if hasattr(http_client, 'close') else None)
            except Exception as e:
                print(f"Failed to initialize shared vLLM client: {e}")
                vllm_client = None

    # Separate auxiliary tasks from main tasks for better parallelization
    execution_tasks = []
    llm_judge_tasks = []
    main_tasks = []
    
    for task in chunk_tasks:
        enable_execution_reward, enable_llm_judge_task = task[9], task[10]
        if enable_execution_reward:
            execution_tasks.append(task)
        elif enable_llm_judge_task:
            llm_judge_tasks.append(task)
        main_tasks.append(task)  # All tasks need main scoring

    # Configure thread pools
    try:
        exec_threads = int(os.getenv("NAIVE_LLM_EXEC_THREADS", "32"))
        main_threads = int(os.getenv("NAIVE_LLM_MAIN_THREADS", "8"))
    except Exception:
        exec_threads = 4
        main_threads = 8
    
    exec_threads = max(1, min(exec_threads, len(execution_tasks))) if execution_tasks else 1
    main_threads = max(1, min(main_threads, len(main_tasks)))

    def process_execution_task(task):
        """Process execution rewards."""
        (index, token_strings, response_str, prompt_str, ground_truth, 
         data_source, extra_info, compute_score_fn, score_source, enable_execution_reward, 
         enable_llm_judge_task, vllm_api_base, vllm_model, judge_timeout, judge_max_retries) = task
        
        seg_ids = _build_seg_ids_from_tokens(token_strings)
        auxiliary_rewards = [0.0] * len(token_strings)
        stats_dict = {
            "execution": 0.0,
            "tool_use_count": _count_tool_use(response_str),
            "factual_correctness": 0.0,
            "search_efficiency": 0.0,
            "answer_clarity": 0.0,
        }
        
        execution_scores = _compute_execution_scores_from_pairs(response_str)
        auxiliary_rewards = _per_token_execution_rewards(seg_ids, execution_scores)
        if execution_scores:
            stats_dict["execution"] = float(sum(execution_scores) / len(execution_scores))
        
        return index, seg_ids, auxiliary_rewards, stats_dict
    
    def batch_process_llm_judge():
        """Batch process LLM-judge tasks."""
        if not llm_judge_tasks or not vllm_client:
            return {}
        
        # Prepare batch requests
        batch_requests = []
        for task in llm_judge_tasks:
            (index, token_strings, response_str, prompt_str, ground_truth, 
             data_source, extra_info, compute_score_fn, score_source, enable_execution_reward, 
             enable_llm_judge_task, vllm_api_base, vllm_model, judge_timeout, judge_max_retries) = task
            
            tool_pairs = _extract_tool_pairs(response_str)
            if tool_pairs:
                # Extract question
                question = ""
                if isinstance(extra_info, dict):
                    question = extra_info.get("question", "")
                    if not question and "prompt" in extra_info:
                        prompt_content = extra_info["prompt"]
                        if isinstance(prompt_content, str):
                            question = prompt_content
                        elif isinstance(prompt_content, list):
                            for item in prompt_content:
                                if isinstance(item, dict) and item.get('role') == 'user':
                                    question = item.get('content', '')
                                    break
                if not question and isinstance(ground_truth, str):
                    question = ground_truth
                if not question:
                    question = prompt_str
                
                batch_requests.append({
                    'index': index,
                    'question': question,
                    'tool_pairs': tool_pairs,
                    'token_strings': token_strings,
                    'response_str': response_str
                })
        
        # Batch evaluate
        batch_results = _batch_llm_judge_evaluate(
            batch_requests, vllm_client, vllm_model, judge_timeout, judge_max_retries
        )
        
        # Process results and create auxiliary rewards
        llm_judge_results = {}
        for result in batch_results:
            index = result['index']
            score_triplets = result['score_triplets']
            
            # Find corresponding task data
            task_data = None
            for req in batch_requests:
                if req['index'] == index:
                    task_data = req
                    break
            
            if not task_data:
                continue
                
            token_strings = task_data['token_strings']
            response_str = task_data['response_str']
            seg_ids = _build_seg_ids_from_tokens(token_strings)
            auxiliary_rewards = [0.0] * len(token_strings)
            
            stats_dict = {
                "execution": 0.0,
                "tool_use_count": _count_tool_use(response_str),
                "factual_correctness": 0.0,
                "search_efficiency": 0.0,
                "answer_clarity": 0.0,
            }
            
            if score_triplets:
                import numpy as np
                ps = np.array(score_triplets, dtype=float)
                fc_avg, se_avg, ac_avg = ps.mean(axis=0).tolist()
                stats_dict.update({
                    "factual_correctness": fc_avg,
                    "search_efficiency": se_avg,
                    "answer_clarity": ac_avg,
                })
                
                # Convert to auxiliary rewards
                coeffs = (0.15, 0.15, 0.15)
                auxiliary_scores = [
                    coeffs[0] * fc + coeffs[1] * se + coeffs[2] * ac 
                    for fc, se, ac in score_triplets
                ]
                
                if auxiliary_scores:
                    boundaries = _compute_segment_boundaries(seg_ids)
                    if boundaries:
                        per_segment_auxiliary = [0.0] * len(boundaries)
                        for i, score in enumerate(auxiliary_scores[:len(boundaries)]):
                            per_segment_auxiliary[i] = float(score)
                        if len(auxiliary_scores) > len(boundaries):
                            per_segment_auxiliary[-1] += sum(
                                float(v) for v in auxiliary_scores[len(boundaries):]
                            )
                        
                        start_idx = 0
                        for seg_idx, end_idx in enumerate(boundaries):
                            seg_reward = per_segment_auxiliary[seg_idx]
                            for pos in range(start_idx, end_idx + 1):
                                auxiliary_rewards[pos] += seg_reward
                            start_idx = end_idx + 1
            
            llm_judge_results[index] = (index, seg_ids, auxiliary_rewards, stats_dict)
        
        return llm_judge_results

    def process_main_task(task):
        """Process main compute_score task."""
        (index, token_strings, response_str, prompt_str, ground_truth, 
         data_source, extra_info, compute_score_fn, score_source, *_) = task
        
        main_score = 0.0
        main_extra = {}
        stats_dict = {}

        if compute_score_fn:
            try:
                score = compute_score_fn(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    score_source=score_source,
                )
                if isinstance(score, dict):
                    main_score = float(score.get("score", 0.0))
                    main_extra = {k: v for k, v in score.items()}
                    stats_dict["f1"] = float(score.get("f1", 0.0))
                    stats_dict["em"] = float(score.get("em", 0.0))
                    stats_dict["format_reward"] = float(score.get("format_reward", 0.0))
                else:
                    main_score = float(score)
                    stats_dict["f1"] = 0.0
                    stats_dict["em"] = 0.0
                    stats_dict["format_reward"] = 0.0
            except Exception as e:
                main_score = 0.0
                main_extra = {"error": str(e)}
                stats_dict["f1"] = 0.0
                stats_dict["em"] = 0.0
                stats_dict["format_reward"] = 0.0
        
        return index, main_score, main_extra, stats_dict

    # Process auxiliary tasks (execution and LLM-judge) and main tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=exec_threads) as exec_pool, \
         concurrent.futures.ThreadPoolExecutor(max_workers=main_threads) as main_pool:
        
        # Submit execution tasks
        exec_futures = []
        if execution_tasks:
            exec_futures = [exec_pool.submit(process_execution_task, task) for task in execution_tasks]
        
        # Process LLM-judge tasks in batch (runs in main thread for simplicity)
        if llm_judge_tasks:
            llm_judge_results = batch_process_llm_judge()
        else:
            llm_judge_results = {}
        
        # Submit main tasks
        main_futures = [main_pool.submit(process_main_task, task) for task in main_tasks]
        
        # Collect execution results
        for fut in concurrent.futures.as_completed(exec_futures):
            idx, seg_ids, auxiliary_rewards, exec_stats = fut.result()
            seg_ids_by_index[idx] = seg_ids
            auxiliary_rewards_by_index[idx] = auxiliary_rewards
            # Merge execution stats
            if idx not in stats_by_index:
                stats_by_index[idx] = {}
            stats_by_index[idx].update(exec_stats)
        
        # Collect LLM-judge results
        if llm_judge_tasks:
            for idx, (_, seg_ids, auxiliary_rewards, judge_stats) in llm_judge_results.items():
                seg_ids_by_index[idx] = seg_ids
                auxiliary_rewards_by_index[idx] = auxiliary_rewards
                # Merge judge stats
                if idx not in stats_by_index:
                    stats_by_index[idx] = {}
                stats_by_index[idx].update(judge_stats)
        
        # Collect main results
        for fut in concurrent.futures.as_completed(main_futures):
            idx, main_score, main_extra, main_stats = fut.result()
            main_score_by_index[idx] = main_score
            main_extra_by_index[idx] = main_extra
            # Merge main stats
            if idx not in stats_by_index:
                stats_by_index[idx] = {}
            stats_by_index[idx].update(main_stats)

    return seg_ids_by_index, auxiliary_rewards_by_index, main_score_by_index, main_extra_by_index, stats_by_index


@register("naive_llm")
class NaiveLLMRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, 
    compute_score=None, 
    eval=False, 
    reward_fn_key="data_source", 
    num_processes: int = 48, 
    enable_execution_reward: bool = False,
    enable_process_evaluation: bool = False, 
    enable_llm_judge: Any = None, 
    vllm_api_base: Any = None, 
    vllm_model: str = "qwen2.5-7b-instruct", 
    judge_timeout: float = 30.0, 
    judge_max_retries: int = 3, 
    score_source="em") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        # Auto-detect optimal process count if not specified
        if num_processes is None:
            import multiprocessing
            self.num_processes = max(1, min(multiprocessing.cpu_count(), 8))
        else:
            self.num_processes = max(1, int(num_processes))
        self.enable_execution_reward = bool(enable_execution_reward)
        self.enable_process_evaluation = bool(enable_process_evaluation)
        print(f"enable_execution_reward: {self.enable_execution_reward}, enable_process_evaluation: {self.enable_process_evaluation}"
        )
        # If enable_llm_judge is None, fall back to enable_process_evaluation for backward compat
        self.enable_llm_judge = bool(enable_process_evaluation)
        self.vllm_model = vllm_model
        self.judge_timeout = float(judge_timeout)
        self.judge_max_retries = int(judge_max_retries)
        self.score_source = score_source
        # Store vLLM API base for worker initialization (avoid client serialization issues)
        self.vllm_api_base = "http://10.24.1.118:9000/v1"
        if self.enable_llm_judge and OpenAI is not None:
            self.vllm_api_base = vllm_api_base or ""
            if self.vllm_api_base:
                print(f"LLM-as-judge enabled with vLLM at {self.vllm_api_base}")
            else:
                print("Warning: LLM-as-judge enabled but no vLLM API base configured")
                self.enable_llm_judge = False

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        # Create reward_tensor with shape (bs, T+1) where last column is for final reward
        responses_shape = data.batch['responses'].shape  # (bs, T)
        bs, T = responses_shape
        reward_tensor = torch.zeros((bs, T + 1), dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # Create stats tensor (like code.py): [format_reward, acc, execution, tool_use_count, factual_correctness, search_efficiency, answer_clarity]
        stats_tensor = torch.zeros((bs, 8), dtype=torch.float32)

        already_print_data_sources = {}

        # Pre-process data and prepare for batch decoding
        batch_valid_prompt_ids = []
        batch_valid_response_ids = []
        batch_metadata = []  # Store metadata for each sample
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]
            
            batch_valid_prompt_ids.append(valid_prompt_ids.tolist())
            batch_valid_response_ids.append(valid_response_ids.tolist())
            
            # Prepare metadata
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            batch_metadata.append({
                'index': i,
                'ground_truth': ground_truth,
                'data_source': data_source,
                'extra_info': extra_info,
                'valid_response_length': valid_response_length,
            })

        # Batch decode all prompts and responses
        if len(batch_valid_prompt_ids) > 1:
            print(f"Batch decoding {len(batch_valid_prompt_ids)} prompts and responses...")
        try:
            # Use batch_decode for efficiency
            if hasattr(self.tokenizer, 'batch_decode'):
                prompt_strs_list = self.tokenizer.batch_decode(batch_valid_prompt_ids, skip_special_tokens=True)
                response_strs_list = self.tokenizer.batch_decode(batch_valid_response_ids, skip_special_tokens=False)
            else:
                # Fallback to individual decode if batch_decode not available
                prompt_strs_list = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch_valid_prompt_ids]
                response_strs_list = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in batch_valid_response_ids]
        except Exception as e:
            print(f"Batch decode failed, falling back to individual decode: {e}")
            prompt_strs_list = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch_valid_prompt_ids]
            response_strs_list = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in batch_valid_response_ids]

        # Batch convert tokens for segment detection
        if len(batch_valid_response_ids) > 1:
            print(f"Batch converting {len(batch_valid_response_ids)} token sequences...")
        batch_token_strings = []
        try:
            if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                for response_ids in batch_valid_response_ids:
                    converted = self.tokenizer.convert_ids_to_tokens(response_ids)
                    token_strings = [t if isinstance(t, str) else "" for t in converted]
                    batch_token_strings.append(token_strings)
            else:
                raise AttributeError("convert_ids_to_tokens not available")
        except Exception as e:
            print(f"Batch token conversion failed, falling back to individual decode: {e}")
            for response_ids in batch_valid_response_ids:
                token_strings = [self.tokenizer.decode([int(t)], skip_special_tokens=False) for t in response_ids]
                batch_token_strings.append(token_strings)

        # Prepare tasks with batch-decoded data
        tasks: List[Tuple[int, List[str], str, str, Any, str, dict, Any, bool, bool, Any, float, int, Any]] = []
        prompt_strs: dict[int, str] = {}
        response_strs: dict[int, str] = {}
        ground_truths: dict[int, Any] = {}
        data_sources: dict[int, Any] = {}
        token_strings_by_index: dict[int, List[str]] = {}
        
        for idx, (prompt_str, response_str, token_strings, metadata) in enumerate(
            zip(prompt_strs_list, response_strs_list, batch_token_strings, batch_metadata)
        ):
            i = metadata['index']
            prompt_strs[i] = prompt_str
            response_strs[i] = response_str
            ground_truths[i] = metadata['ground_truth']
            data_sources[i] = metadata['data_source']
            token_strings_by_index[i] = token_strings
            
            # assemble worker args
            tasks.append((
                i,
                token_strings,
                response_str,
                prompt_str,
                metadata['ground_truth'],
                metadata['data_source'],
                metadata['extra_info'],
                self.compute_score,
                self.score_source,
                self.enable_execution_reward,
                self.enable_llm_judge,
                self.vllm_api_base,
                self.vllm_model,
                self.judge_timeout,
                self.judge_max_retries,
            ))

        # Run process pool to compute auxiliary rewards, main scores, and stats
        auxiliary_rewards_by_index: dict[int, List[float]] = {}
        seg_ids_by_index: dict[int, List[int]] = {}
        main_score_by_index: dict[int, float] = {}
        main_extra_by_index: dict[int, dict] = {}
        stats_by_index: dict[int, dict] = {}
        # Use ProcessPoolExecutor for parallel processing or when special features are needed
        use_process_pool = (self.num_processes > 1 or self.enable_execution_reward or self.enable_llm_judge)
        
        if use_process_pool:
            n_tasks = len(tasks)
            if n_tasks > 0:
                num_workers = max(1, min(self.num_processes, n_tasks))
                chunk_size = (n_tasks + num_workers - 1) // num_workers  # ceil division
                chunks = [tasks[i:i + chunk_size] for i in range(0, n_tasks, chunk_size)]

                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(_process_chunk_worker, ch) for ch in chunks]
                    for fut in concurrent.futures.as_completed(futures):
                        (seg_chunk,
                         aux_chunk,
                         main_score_chunk,
                         main_extra_chunk,
                         stats_chunk) = fut.result()
                        seg_ids_by_index.update(seg_chunk)
                        auxiliary_rewards_by_index.update(aux_chunk)
                        main_score_by_index.update(main_score_chunk)
                        main_extra_by_index.update(main_extra_chunk)
                        stats_by_index.update(stats_chunk)
        else:
            # Fallback: compute seg_ids and basic scores without multiprocessing
            # This is needed for SGRPO and basic reward computation
            (seg_ids_by_index,
             auxiliary_rewards_by_index,
             main_score_by_index,
             main_extra_by_index,
             stats_by_index) = _process_chunk_worker(tasks)

        # Aggregate rewards
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())

            # Add auxiliary rewards (execution or LLM-as-judge) per token to sequence positions
            auxiliary_rewards = auxiliary_rewards_by_index.get(i)
            if auxiliary_rewards:
                # Ensure we don't exceed tensor or auxiliary rewards bounds
                max_pos = min(len(auxiliary_rewards), valid_response_length, reward_tensor.shape[1] - 1)
                for pos in range(max_pos):
                    reward_tensor[i, pos] += float(auxiliary_rewards[pos])

            # Main score goes to the final position (T+1)
            main_score = main_score_by_index.get(i, 0.0)
            main_extra = main_extra_by_index.get(i, {})
            
            # Final reward goes to the last column (T+1)
            reward_tensor[i, -1] = float(main_score)

            # Store extra info from main compute_score
            if isinstance(main_extra, dict):
                for key, value in main_extra.items():
                    reward_extra_info[key].append(value)
            else:
                # Record at least the scalar score
                reward_extra_info["score"].append(main_score)

            # Populate stats tensor (like code.py)
            stats_dict = stats_by_index.get(i, {})
            stats_tensor[i, 0] = float(stats_dict.get("format_reward", 0.2))
            stats_tensor[i, 1] = float(stats_dict.get("f1", 0.0))
            stats_tensor[i, 2] = float(stats_dict.get("em", 0.0))
            stats_tensor[i, 3] = float(stats_dict.get("execution", 0.0))
            stats_tensor[i, 4] = float(stats_dict.get("tool_use_count", 0))
            stats_tensor[i, 5] = float(stats_dict.get("factual_correctness", 0.0))
            stats_tensor[i, 6] = float(stats_dict.get("search_efficiency", 0.0))
            stats_tensor[i, 7] = float(stats_dict.get("answer_clarity", 0.0))


            prompt_str = prompt_strs[i]
            response_str = response_strs[i]
            ground_truth = ground_truths[i]
            data_source = data_sources[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # if already_print_data_sources[data_source] < self.num_examine:
            #     already_print_data_sources[data_source] += 1
            #     print("[prompt]", prompt_str)
            #     print("[response]", response_str)
            #     print("[ground_truth]", ground_truth)
            #     print("[main_score]", main_score)
            #     if isinstance(main_extra, dict) and main_extra:
            #         for key, value in main_extra.items():
            #             print(f"[{key}]", value)

        # Save sample data for debugging and analysis (similar to code.py)
        # try:
        #     _save_sample_data(
        #         data, reward_tensor, main_extra_by_index, auxiliary_rewards_by_index, 
        #         seg_ids_by_index, self.tokenizer, max_samples=3
        #     )
        # except Exception as e:
        #     print(f"Warning: Failed to save sample data: {e}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "stats_tensor": stats_tensor,
                "seg_ids_by_index": seg_ids_by_index,
            }
        else:
            # Return both reward and stats tensors (like code.py)
            return reward_tensor, stats_tensor
