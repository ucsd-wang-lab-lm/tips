import torch
import os
import concurrent.futures
import requests
import re
import json
from verl import DataProto
from typing import List, Tuple, Dict, Any
import numpy as np
import time
# Add OpenAI import for vLLM process evaluation
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("Warning: OpenAI package not found. Process evaluation will be disabled.")

CODE_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Process evaluation system prompt template
PROCESS_EVAL_SYSTEM_PROMPT = """
You are an evaluation assistant. Your goal is to assess how well a large language model used Jupyter Notebook to complete a code generation (codegen) task.

Evaluate each <execute>…</execute> and its following <results>…</results> pair in the user content (in order of appearance after the heading "### Execution of Jupyter notebook by the user:"). For EACH pair, score the following three dimensions on a 0–5 Likert scale (integers):
- run_correct (execution correctness):
  0: cannot run/fatal errors; no valid result
  1: runs barely; mostly fails/clearly wrong results
  2: runs but many errors; only a few cases correct
  3: mostly correct; occasional errors or weak edge handling
  4: all visible cases correct; minor robustness/exception gaps
  5: robust and correct; edge cases and exceptions handled properly
- on_topic (topical alignment; separate from tool/execution compliance):
  0: largely unrelated to the task
  1: mostly off-topic; few relevant parts
  2: partially relevant but mixed with substantial irrelevant content
  3: mainly on-topic with small deviations or verbosity
  4: tightly on-topic with negligible deviations
  5: strictly focused on the task and requirements
- good_tests (test design quality):
  0: no tests or unexecutable tests
  1: single example; no verification (prints only)
  2: few cases; limited coverage; lacks edge cases/assertions
  3: diverse cases with basic assertions; average edge coverage
  4: systematic coverage of main paths and edges with clear assertions
  5: high-quality suite covering normal/edge/negative/random cases; reproducible and diagnostic

Output requirements:
- First provide your reasoning as a concise, structured chain-of-thought (cot) that cites concrete evidence from EACH pair. Do not omit key clues. Clearly separate reasoning per pair (e.g., Pair 1, Pair 2, …).
- Then provide the final answer as a SINGLE line containing a list of ratings for all pairs in order. Each pair contributes a triplet [run_correct, on_topic, good_tests]. If no pairs are present, output an empty list [].
- Strictly follow this exact template (preserve tags, order, and casing):

<think>
Your step-by-step analysis per pair…
</think>
<answer>
ratings_by_pair=[[r1_run,r1_topic,r1_tests],[r2_run,r2_topic,r2_tests],...]
</answer>

Additional notes:
- In the <answer> block, output EXACTLY ONE LINE as shown above, starting with "ratings_by_pair=". Do not add any other lines, keys, tags, code fences, or text inside or outside the <answer> block.
- Judge on_topic strictly by topical alignment to the task. Do not mark on_topic low due to execution/tool-usage non-compliance (e.g., stdin usage, exceeding tool-call budget). Reflect such non-compliance in run_correct and in the reasoning, not in on_topic.
- If evidence is insufficient for a pair, explicitly note the uncertainty in your reasoning and choose the closest rating by anchors.
- Do not output any additional structured fields outside the answer block.
"""
def extract_code_from_answer_block(text: str) -> str:
    """
    Strictly extract code from the last <answer>...</answer>.
    If no <answer> or no code block inside, return "".
    """
    matches = ANSWER_PATTERN.findall(text or "")
    if not matches:
        return ""
    answer_text = matches[-1].strip()
    code_blocks = CODE_PATTERN.findall(answer_text)
    if code_blocks:
        return code_blocks[-1].strip()
    return ""

def try_extract_solution(solution_str: str) -> str:
    """
    Finds all <answer> blocks and returns the content of the last one.
    If no <answer> block is found, returns the original string for compatibility.
    """
    matches = ANSWER_PATTERN.findall(solution_str)
    
    if matches:
        return matches[-1].strip()
    
    return solution_str

def _is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except (TypeError, OverflowError):
        return False

def count_num_tool_use(solution_str) -> int:
    """Count the number of tool uses in the solution string."""
    pattern = r"<execute>(.*?)</execute>"
    matches = re.findall(pattern, solution_str, re.DOTALL)
    return len(matches)

def extract_code_from_string(solution_str):
    solution_str = try_extract_solution(solution_str)
    code_blocks = CODE_PATTERN.findall(solution_str)
    if code_blocks:
        return code_blocks[-1].strip()
    return ""

def extract_question_key(text: str) -> str:
    """
    Extract text between "Question:\n" and "\n\nThe test cases are some" using regex.
    
    Args:
        text: The input text to search in
        
    Returns:
        The extracted text between the specified patterns, or None if not found
    """
    # Pattern to match text between "Question:\n" and "\n\nThe test cases are some"
    pattern = r'Question:\n(.*?)\n\nThe test cases are some'
    
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_question_from_prompt(prompt_data) -> str:
    """
    Extract question text from prompt data which can be either a string or a list of dicts.
    
    Args:
        prompt_data: Either a string or a list of dictionaries with 'role' and 'content' fields
        
    Returns:
        The question text, or None if not found
    """
    if isinstance(prompt_data, str):
        return prompt_data
    elif isinstance(prompt_data, list):
        # Look for user message in the prompt list
        for item in prompt_data:
            if isinstance(item, dict) and item.get('role') == 'user':
                content = item.get('content', '')
                if isinstance(content, str):
                    return content
    return None

def extract_execute_results_pairs(text: str) -> List[str]:
    """Extract <execute>...</execute> and <results>...</results> pairs from text."""
    pattern = re.compile(r"<execute>.*?</execute>.*?<results>.*?</results>", re.DOTALL | re.IGNORECASE)
    return [m.group(0).strip() for m in pattern.finditer(text or "")]

# New: compute per-pair execution scores based on <results> content
def compute_execution_scores_from_pairs(text: str) -> List[float]:
    pairs = extract_execute_results_pairs(text or "")
    scores: List[float] = []
    results_pattern = re.compile(r"<results>(.*?)</results>", re.DOTALL | re.IGNORECASE)

    # Benign suffix lines to strip before judging emptiness
    benign_suffix_patterns = [
        re.compile(r"^\s*Tool calls remaining:\s*-?\d+\s*$", re.IGNORECASE),
        re.compile(r"^\s*Run Time.*$", re.IGNORECASE),
    ]

    # Strong negative signals (case-insensitive)
    negative_keywords = [
        # generic
        r"error", r"exception", r"traceback",
        # time/memory/resource
        r"timeout", r"time limit", r"time-limit", r"killed", r"memoryerror", r"memory limit", r"memory_violation",
        # syntax/compile/runtime
        r"syntaxerror", r"indentationerror", r"invalid syntax", r"compilation_error", r"runtime_error",
        # server/worker
        r"worker terminated", r"process pool", r"terminated abruptly", r"pool crashed", r"operation blocked", r"restricted",
        # tool orchestration
        r"no chance left", r"tool call timeout", r"tool call failed", r"no code to execute",
        # sql
        r"sql error", r"error executing query",
        # scoring labels
        r"output_mismatch", r"unknown_error",
    ]
    negative_re = re.compile(r"(" + r"|".join(negative_keywords) + r")", re.IGNORECASE)

    # Lines that if they are the only content should be treated as non-signal
    ignorable_only_lines = [
        re.compile(r"^\s*$"),
    ]

    for pair in pairs:
        m = results_pattern.search(pair)
        if not m:
            scores.append(0.0)
            continue

        results_body = m.group(1)
        # Normalize newlines and trim
        lines = [ln.rstrip() for ln in (results_body or "").splitlines()]
        # Strip benign suffix lines like "Tool calls remaining: X" and "Run Time..."
        cleaned_lines: List[str] = []
        for ln in lines:
            if any(pat.match(ln) for pat in benign_suffix_patterns):
                continue
            cleaned_lines.append(ln)
        cleaned_text = "\n".join(cleaned_lines).strip()

        # If after cleanup there is no meaningful content, mark as success (empty results are OK)
        if not cleaned_text or all(pat.match(cleaned_text) for pat in ignorable_only_lines):
            scores.append(0.0)
            continue

        # If any negative keyword appears, mark as failure
        if negative_re.search(cleaned_text):
            scores.append(0.0)
            continue

        # Otherwise treat as success
        scores.append(0.5)

    return scores

def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
    """Build message list for OpenAI API."""
    messages: List[Dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages

class CodeRewardManager:
    """
    A thread-based reward manager that computes code-based
    rewards in parallel via the code server and optionally
    provides process evaluation using vLLM.
    """

    def __init__(self, tokenizer, num_workers=None, server_url="http://localhost:5001", 
                 config=None, vllm_api_base=None, eval_mode=True,
                 vllm_model="qwen3-4B", vllm_temperature=0.0, vllm_max_tokens=6024):
        self.tokenizer = tokenizer
        self.server_url = server_url
        self.max_workers = num_workers or min(32, (os.cpu_count() or 1) * 4)
        self.config = config
        
        self.enable_process_eval = self.config.get("enable_process_eval", False) and OpenAI is not None
        self.enable_execution_reward = self.config.get("enable_execution_reward", False)
        if eval_mode:
            self.enable_process_eval = False
            self.enable_execution_reward = False

        # vLLM client settings for process evaluation
        self.vllm_client = None
        self.vllm_model = vllm_model
        self.vllm_temperature = vllm_temperature
        self.vllm_max_tokens = vllm_max_tokens
        # Coefficients for mapping (run_correct, on_topic, good_tests) -> scalar
        self.process_coeffs = (0.2, 0.2, 0.2)
        
        if self.enable_process_eval:
            vllm_api_base = os.getenv("VLLM_API_BASE", "http://10.24.0.38:8000/v1")  # Default vLLM API base if not set in env
            if not vllm_api_base:
                raise EnvironmentError("VLLM_API_BASE not set")
            try:
                self.vllm_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
                    base_url=vllm_api_base,
                )
                print(f"Process evaluation enabled with vLLM at {vllm_api_base}")
            except Exception as e:
                print(f"Failed to initialize vLLM client: {e}")
                self.enable_process_eval = False

        # Check if code server is reachable
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5.0)
            if response.status_code != 200:
                raise RuntimeError(f"Code server at {self.server_url} returned status code {response.status_code}")
            print(f"Successfully connected to code server at {self.server_url}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to code server at {self.server_url}: {str(e)}")

    def compute_process_evaluation(self, question_text: str, response_text: str) -> List[Tuple[float, float, float]]:
        """
        Compute process evaluation ratings per execute/results pair using vLLM.
        
        Args:
            question_text: The original question/problem text
            response_text: The model's response containing <execute> and <results> pairs
            
        Returns:
            List of triplets (run_correct, on_topic, good_tests) normalized to [0, 1], one per pair
        """
        if not self.enable_process_eval or not self.vllm_client:
            return []
        
        try:
            # Extract execute-results pairs from response
            execute_pairs = extract_execute_results_pairs(response_text)
            
            if not execute_pairs:
                return []
            
            # Build user prompt
            user_prompt = f"Question:\n{question_text}\n\n### Execution of Jupyter notebook by the user:\n\n"
            user_prompt += "\n\n".join(execute_pairs) + "/no_think"
            
            # Build messages
            messages = build_messages(PROCESS_EVAL_SYSTEM_PROMPT, user_prompt)

            # print(f"DEBUG: messages: {messages}")
            
            # Call vLLM API
            completion = self.vllm_client.chat.completions.create(
                model=self.vllm_model,
                messages=messages,
                temperature=self.vllm_temperature,
                max_tokens=self.vllm_max_tokens
            )
            
            content = completion.choices[0].message.content
            log = ""
            log = f"DEBUG: content: {content}\n"
            # print(log)
            if content is None:
                return []
            
            # Parse the evaluation result into per-pair ratings
            import ast
            
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
                    rc, ot, gt = float(item[0]) / 5.0, float(item[1]) / 5.0, float(item[2]) / 5.0
                    normalized.append((rc, ot, gt))
                except Exception:
                    continue
            # log = f"DEBUG: normalized: {normalized}\n"
            # log += "----------------------------------------\n\n"
            # print(log)
            return normalized
        
        except Exception as e:
            print(f"Error in process evaluation: {e}")
            return []

    def compute_single_score(self, task):
        i, candidate_code, test_case, extra_info, config, resp_token_ids, seg_ids_arr = task

        # --- New: strictly use code from the final answer segment (seg_id == -1); no fallback ---
        final_answer_code = ""
        if seg_ids_arr is not None:
            neg1_positions = np.where(seg_ids_arr == -1)[0] if hasattr(seg_ids_arr, "dtype") else [idx for idx, v in enumerate(seg_ids_arr) if v == -1]
            if len(neg1_positions) > 0:
                start = int(neg1_positions[0])
                end = int(neg1_positions[-1]) + 1
                final_segment_tokens = resp_token_ids[start:end]
                final_segment_text = self.tokenizer.decode(final_segment_tokens)
                final_answer_code = extract_code_from_answer_block(final_segment_text)

        candidate_code_clean = final_answer_code
        num_tool_use = count_num_tool_use(candidate_code)

        # Initialize scores
        format_reward = 1.0 if candidate_code_clean != "" else 0.0
        acc = 0.0
        execution = 0.0
        process_scores: List[Tuple[float, float, float]] = []
        execution_scores = []
        
        if isinstance(test_case, dict):
            test_case_str = json.dumps(test_case)
        else:
            test_case_str = test_case

        if isinstance(extra_info, dict):
            extra_info_str = json.dumps(extra_info)
        else:
            extra_info_str = extra_info

        # Compute execution scores per <execute>/<results> pair and scalarize by average
        execution_scores = []  # Initialize as empty list to prevent NameError
        if self.enable_execution_reward:
            execution_scores = compute_execution_scores_from_pairs(candidate_code)
            if execution_scores:
                execution = float(sum(execution_scores) / len(execution_scores))
            else:
                execution = 0.0

        log = ""
        log += f"DEBUG: original response: {candidate_code}\n\n "
        log += f"DEBUG: enable_execution_reward: {self.enable_execution_reward}\n"
        log += f"DEBUG: enable_process_eval: {self.enable_process_eval}\n"
        log += f"DEBUG: execution_scores: {execution_scores}\n\n"
        log += "----------------------------------------\n\n"

        # Define code server execution function
        def execute_code_server():
            nonlocal acc
            if candidate_code_clean != "":
                payload = {
                    "candidate_code": candidate_code_clean,
                    "test_case": test_case_str,
                    "extra_info": extra_info_str
                }

                # Send to code server
                max_retries = 3
                retry_count = 0
                execution_successful = False

                while retry_count <= max_retries and not execution_successful:
                    try:
                        resp = requests.post(
                            f"{self.server_url}/code_test",
                            json=payload,
                            timeout=500,
                        )
                        
                        if resp.status_code == 200:
                            data = resp.json()
                            score = data.get("score", 0.0)
                            
                            # Check if error contains "worker terminated" and we should retry
                            response_text = resp.text.lower()
                            if (score == 0.0 and retry_count < max_retries and 
                                ("worker terminated" in response_text or 
                                 "process pool" in response_text or
                                 "terminated abruptly" in response_text or
                                 "pool crashed" in response_text)):
                                retry_count += 1
                                print(f"[idx={i}] Worker error detected, retrying ({retry_count}/{max_retries})")
                                time.sleep(1)  # Brief delay before retry
                                continue
                            
                            # Success or non-retryable error
                            if retry_count > 0:
                                print(f"[idx={i}] Succeeded after {retry_count} retries, score={score}")
                            
                            acc = score
                            execution_successful = True
                            
                        else:
                            print(f"[idx={i}] Server returned status {resp.status_code}: {resp.text[:200]}")
                            if retry_count < max_retries:
                                retry_count += 1
                                time.sleep(1)
                                continue
                            else:
                                execution_successful = True
                                
                    except requests.exceptions.Timeout as e:
                        if retry_count < max_retries:
                            retry_count += 1
                            print(f"[idx={i}] Timeout error, retrying ({retry_count}/{max_retries})")
                            time.sleep(1)
                            continue
                        else:
                            print(f"[idx={i}] Timeout after {max_retries} retries: {e}")
                            execution_successful = True
                            
                    except requests.exceptions.ConnectionError as e:
                        if retry_count < max_retries:
                            retry_count += 1
                            print(f"[idx={i}] Connection error, retrying ({retry_count}/{max_retries})")
                            time.sleep(1)
                            continue
                        else:
                            print(f"[idx={i}] Connection error after {max_retries} retries: {e}")
                            execution_successful = True
                            
                    except Exception as e:
                        print(f"[idx={i}] Unexpected error: {e}")
                        execution_successful = True
        
        # Define vLLM process evaluation function
        def execute_process_eval():
            nonlocal process_scores
            if self.enable_process_eval:
                if isinstance(extra_info, str):
                    extra_info_parsed = json.loads(extra_info)
                else:
                    extra_info_parsed = extra_info
                    
                # Extract question from extra_info or test_case
                question_text = ""
                if isinstance(extra_info_parsed, dict):
                    question_text = extra_info_parsed.get("question", "")
                    if not question_text and "prompt" in extra_info_parsed:
                        question_text = extract_question_from_prompt(extra_info_parsed["prompt"])
                    elif "original_prompt" in extra_info_parsed:
                        question_text = extra_info_parsed["original_prompt"]

                if question_text:
                    process_scores = self.compute_process_evaluation(question_text, candidate_code)
        
        # Execute both tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            code_future = executor.submit(execute_code_server)
            if self.enable_process_eval:
                process_future = executor.submit(execute_process_eval)
            
            # Wait for both to complete
            if self.enable_process_eval:
                concurrent.futures.wait([code_future, process_future])
                # print(f"DEBUG: process_scores: {process_scores}")
            else:
                concurrent.futures.wait([code_future])
        
        # Combine all scores: execution stats and process eval scores list
        return (i, (format_reward, acc, execution, num_tool_use, process_scores, execution_scores))

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # Create reward_tensor with shape (bs, T+1) where last column is for final reward
        responses_shape = data.batch['responses'].shape  # (bs, T)
        bs, T = responses_shape
        reward_tensor = torch.zeros((bs, T + 1), dtype=torch.float32)
        
        N = data.batch['responses'].shape[0]
        # Expand stats tensor to include process evaluation scores: [format_reward, acc, eff, tool_use_count, run_correct, on_topic, good_tests]
        stats_tensor = torch.zeros((N, 7), dtype=torch.float32)

        tasks = []
        idx_to_resp_len = {}

        # Extract data and create tasks
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode response to text
            sequences = valid_response_ids
            candidate_code = self.tokenizer.decode(sequences)
            test_case = data_item.non_tensor_batch.get('test_cases', {})
            extra_info = data_item.non_tensor_batch.get('extra_info', {})

            # Collect the per-token arrays we need in compute_single_score
            resp_token_ids = valid_response_ids.cpu().tolist()  # list[int]
            seg_ids_arr = None
            if 'seg_ids' in data_item.batch:
                seg_ids_arr = data_item.batch['seg_ids'][:valid_response_length].cpu().numpy()
                
            tasks.append((i, candidate_code, test_case, extra_info, self.config, resp_token_ids, seg_ids_arr))

            idx_to_resp_len[i] = valid_response_length.item()

        # Process tasks using ThreadPoolExecutor for parallel execution
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.compute_single_score, task): task[0] for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                except Exception as e:
                    i = futures[future]
                    print(f"Error processing task {i}: {str(e)}")
                    # Return default scores for all metrics: (format_reward, acc, execution, tool_use_count, run_correct, on_topic, good_tests, execution_scores)
                    results.append((i, (0.0, 0.0, 0.0, 0.0, [], [])))

        # Process results and populate tensors
        all_process_scores = {}  # Store process scores for each sample
        for i, (format_reward, acc, execution, tool_use_count, process_scores, execution_scores) in results:
            all_process_scores[i] = process_scores  # Store for later use in save function
            resp_len = idx_to_resp_len[i]
            loss_mask = data[i].batch['loss_mask'][:resp_len]
            
            if i == 10:
                print(f"DEBUG: process_scores: {process_scores}")
                print(f"DEBUG: execution_scores: {execution_scores}")
                print(f"DEBUG: acc: {acc}")

            # Get segment IDs if available
            seg_ids = None
            if 'seg_ids' in data[i].batch:
                seg_ids = data[i].batch['seg_ids'][:resp_len].cpu().numpy()
            
            if seg_ids is not None and (self.enable_execution_reward or self.enable_process_eval):
                # Use segment-based reward assignment (for execution rewards and/or process evaluation)
                self._assign_segment_based_rewards(
                    reward_tensor, i, resp_len, seg_ids, loss_mask,
                    format_reward, acc, execution, execution_scores,
                    process_scores
                )
            else:
                # Fallback to original method (assign to final position)
                end_indices = []
                for j in range(resp_len - 1):
                    if loss_mask[j] == 0 and loss_mask[j + 1] == 1:
                        end_indices.append(j)
                if loss_mask[-1] == 0:
                    end_indices.append(resp_len - 1)

                if end_indices:
                    final_idx = end_indices[-1]
                else:
                    final_idx = resp_len - 1
                
                # Combine rewards: format + accuracy (fallback path, seg_ids always available in practice)
                total_reward = format_reward + acc
                
                # Assign combined reward to final position
                reward_tensor[i, final_idx] += total_reward
            
            # Final reward goes to the last column (T+1)
            final_reward = format_reward + acc
            reward_tensor[i, -1] = final_reward  # Last column for final reward
            
            reward_log = ""
            reward_log += f"format_reward: {format_reward}\n"
            reward_log += f"acc: {acc}\n"
            reward_log += f"execution: {execution}\n"
            reward_log += f"tool_use_count: {tool_use_count}\n"
            reward_log += f"process_scores(pairs): {process_scores}\n"
            reward_log += f"execution_scores: {execution_scores}\n"
            reward_log += f"final_reward: {final_reward}\n"
            # print(reward_log) if i == 0 else None

            if self.enable_process_eval and process_scores:
                ps = np.array(process_scores, dtype=float)  # shape: (K, 3)
                rc_avg, ot_avg, contrib_avg = ps.mean(axis=0).tolist()
            else:
                rc_avg = ot_avg = contrib_avg = 0.0

            if not self.enable_execution_reward:
                execution = 0.0

            # Populate all statistics
            stats_tensor[i, 0] = format_reward
            stats_tensor[i, 1] = acc
            stats_tensor[i, 2] = execution  # Use execution (mean) instead of execution_scores (list)
            stats_tensor[i, 3] = tool_use_count
            stats_tensor[i, 4] = rc_avg
            stats_tensor[i, 5] = ot_avg
            stats_tensor[i, 6] = contrib_avg
            # stats_tensor[i, 4:7] are reserved for process eval if needed

        # Save sample data for debugging and analysis (only for multi-segment samples)
        # self._save_sample_data_if_multi_segment(data, reward_tensor, stats_tensor, all_process_scores)

        return reward_tensor, stats_tensor

    def _build_valid_segment_boundaries(self, seg_ids, resp_len):
        """
        返回所有 seg_id >= 0 的连续 run 的结束位置（end index）。
        -1 段（最终段）不会被合并进来。
        """
        if resp_len <= 0:
            return []
        boundaries = []
        in_valid = (seg_ids[0] >= 0)
        for j in range(1, resp_len):
            if seg_ids[j] != seg_ids[j - 1]:
                if in_valid:
                    boundaries.append(j - 1)      # 离开一个有效段，闭合它
                in_valid = (seg_ids[j] >= 0)
        if in_valid:
            boundaries.append(resp_len - 1)        # 结尾仍在有效段，闭合之
        return boundaries

    def _assign_segment_based_rewards(
        self, reward_tensor, sample_idx, resp_len, seg_ids, loss_mask,
        format_reward, acc, execution, execution_scores, process_scores
    ):
        # 1) 仅构造非 -1 段的边界
        segment_boundaries = self._build_valid_segment_boundaries(seg_ids, resp_len)
        num_segments = len(segment_boundaries)
        if num_segments == 0:
            return

        # 2) 计算每个非 -1 段的执行/过程奖励
        per_segment_execution = [0.0] * num_segments
        if self.enable_execution_reward:
            if execution_scores:
                # 顺序分配，不足则按顺序填满；多余的累加到最后一个有效段
                k = min(len(execution_scores), num_segments)
                for i in range(k):
                    per_segment_execution[i] = execution_scores[i]
                if len(execution_scores) > num_segments:
                    per_segment_execution[-1] += sum(execution_scores[num_segments:])
            else:
                # 无逐对分数，用均分/等策略
                avg = (execution / num_segments) if execution > 0 else 0.0
                per_segment_execution = [avg] * num_segments

        per_segment_process = [0.0] * num_segments
        if self.enable_process_eval and process_scores:
            w_rc, w_ot, w_gt = self.process_coeffs
            k = min(len(process_scores), num_segments)
            for i in range(k):
                rc, ot, gt = process_scores[i]
                per_segment_process[i] = float(w_rc * rc + w_ot * ot + w_gt * gt)
            if len(process_scores) > num_segments:
                tail = process_scores[num_segments:]
                per_segment_process[-1] += sum(float(w_rc * rc + w_ot * ot + w_gt * gt) for (rc, ot, gt) in tail)

        # 3) 将段级奖励写入各段 token（如需更平滑，可只在段末写一次，或按 token 数做归一）
        start_idx = 0
        for seg_idx, end_idx in enumerate(segment_boundaries):
            seg_reward = per_segment_execution[seg_idx] + per_segment_process[seg_idx]
            for pos in range(start_idx, min(end_idx + 1, resp_len)):
                reward_tensor[sample_idx, pos] += seg_reward
            start_idx = end_idx + 1


    def _save_sample_data_if_multi_segment(self, data, reward_tensor, stats_tensor, all_process_scores):
        """
        Save detailed sample data for debugging and analysis only if the sample
        contains multiple segments (i.e., seg_ids is not None and has more than one unique segment ID).
        """
        try:
            import os
            import time
            
            # Create output directory if not exists
            output_dir = "reward_samples"
            os.makedirs(output_dir, exist_ok=True)
            
            # Iterate through data to find samples with multiple segments
            for i in reversed(range(len(data))):
                data_item = data[i]
                
                # Check if this sample has seg_ids and multiple segments
                if 'seg_ids' in data_item.batch:
                    response_length = data_item.batch['responses'].shape[-1]
                    seg_ids_tensor = data_item.batch['seg_ids'][:response_length]
                    seg_ids_np = seg_ids_tensor.cpu().numpy()
                    unique_segments = set(seg_ids_np[seg_ids_np >= 0])  # Ignore -1 padding
                    
                    if len(unique_segments) > 1:  # Multi-segment sample found
                        # Use timestamp for unique filename
                        timestamp = int(time.time())
                        filename = f"{output_dir}/reward_sample_multi_seg_{timestamp}.txt"
                        
                        # Get response data
                        response_ids = data_item.batch['responses'].cpu().tolist()
                        valid_response_length = data_item.batch['attention_mask'][data_item.batch['prompts'].shape[-1]:].sum().item()
                        valid_response_ids = response_ids[:valid_response_length]
                        
                        # Get segment IDs
                        seg_ids = data_item.batch['seg_ids'][:valid_response_length].cpu().tolist()
                        
                        # Get loss_mask and attention_mask
                        loss_mask = data_item.batch.get('loss_mask', torch.ones(data_item.batch['responses'].shape[-1]))[:valid_response_length].cpu().tolist()
                        attention_mask = data_item.batch.get('attention_mask', torch.ones(data_item.batch['responses'].shape[-1] + data_item.batch['prompts'].shape[-1]))[data_item.batch['prompts'].shape[-1]:data_item.batch['prompts'].shape[-1] + valid_response_length].cpu().tolist()
                        
                        # Decode tokens
                        decoded_response = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                        
                        # Get individual token strings
                        token_strings = []
                        for token_id in valid_response_ids:
                            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
                            token_strings.append(token_str)
                        
                        # Get rewards
                        sequence_rewards = reward_tensor[i, :valid_response_length].cpu().tolist()
                        final_reward = reward_tensor[i, -1].item()
                        
                        # Get stats
                        stats = stats_tensor[i].cpu().tolist()
                        
                        # Get additional info
                        test_case = data_item.non_tensor_batch.get('test_cases', {})
                        extra_info = data_item.non_tensor_batch.get('extra_info', {})
                        
                        # Get process scores for this sample
                        process_scores_for_sample = all_process_scores.get(i, [])
                        
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write("=== REWARD MANAGER SAMPLE DATA (MULTI-SEGMENT) ===\n")
                            f.write(f"Timestamp: {timestamp}\n")
                            f.write(f"Sample Index: {i}\n")
                            f.write(f"Unique segments: {sorted(unique_segments)}\n\n")
                            
                            f.write("=== DECODED RESPONSE ===\n")
                            f.write(f"Length: {len(valid_response_ids)} tokens\n")
                            f.write(f"Decoded: {decoded_response}\n\n")
                            
                            f.write("=== TOKEN-LEVEL DETAILS ===\n")
                            f.write("Format: idx | token_id | token_string | seg_id | reward | loss_mask | att_mask | masked_reward\n")
                            f.write("-" * 105 + "\n")
                            
                            for j, (token_id, token_str, reward) in enumerate(zip(valid_response_ids, token_strings, sequence_rewards)):
                                seg_id_str = str(seg_ids[j]) if j < len(seg_ids) else "N/A"
                                loss_mask_val = loss_mask[j] if j < len(loss_mask) else "N/A"
                                att_mask_val = attention_mask[j] if j < len(attention_mask) else "N/A"
                                
                                # Calculate masked reward (apply both masks)
                                if (isinstance(loss_mask_val, (int, float)) and isinstance(att_mask_val, (int, float))):
                                    masked_reward = reward * loss_mask_val * att_mask_val
                                else:
                                    masked_reward = "N/A"
                                
                                f.write(f"{j:3d}: {token_id:6d} | {repr(token_str):20s} | {seg_id_str:>5s} | {reward:8.4f} | {loss_mask_val:>9} | {att_mask_val:>8} | {masked_reward:>12}\n")
                            
                            f.write(f"\n=== REWARDS SUMMARY ===\n")
                            non_zero_rewards = [(j, r) for j, r in enumerate(sequence_rewards) if r != 0.0]
                            f.write(f"Non-zero reward positions: {len(non_zero_rewards)}\n")
                            for pos, reward in non_zero_rewards:
                                f.write(f"  Position {pos}: {reward:.4f}\n")
                            f.write(f"Final reward: {final_reward:.4f}\n")
                            
                            f.write(f"\n=== MASK ANALYSIS ===\n")
                            f.write(f"Loss mask: {loss_mask}\n")
                            f.write(f"Attention mask: {attention_mask}\n")
                            
                            # Analyze mask patterns
                            if loss_mask:
                                active_positions = [i for i, mask in enumerate(loss_mask) if mask == 1 or mask == 1.0]
                                inactive_positions = [i for i, mask in enumerate(loss_mask) if mask == 0 or mask == 0.0]
                                f.write(f"Loss mask active positions ({len(active_positions)}): {active_positions[:20]}{'...' if len(active_positions) > 20 else ''}\n")
                                f.write(f"Loss mask inactive positions ({len(inactive_positions)}): {inactive_positions[:20]}{'...' if len(inactive_positions) > 20 else ''}\n")
                            
                            if attention_mask:
                                att_active = [i for i, mask in enumerate(attention_mask) if mask == 1 or mask == 1.0]
                                att_inactive = [i for i, mask in enumerate(attention_mask) if mask == 0 or mask == 0.0]
                                f.write(f"Attention mask active positions ({len(att_active)}): {att_active[:20]}{'...' if len(att_active) > 20 else ''}\n")
                                f.write(f"Attention mask inactive positions ({len(att_inactive)}): {att_inactive[:20]}{'...' if len(att_inactive) > 20 else ''}\n")
                            
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
                            current_seg = seg_ids[0] if seg_ids else 0
                            for j in range(1, len(seg_ids)):
                                if seg_ids[j] != current_seg and seg_ids[j] >= 0:
                                    boundaries.append(j - 1)
                                    current_seg = seg_ids[j]
                            boundaries.append(len(seg_ids) - 1)
                            
                            f.write(f"Segment boundaries (end positions): {boundaries}\n")
                            
                            # Show rewards at boundaries
                            f.write(f"Rewards at segment boundaries:\n")
                            for j, boundary in enumerate(boundaries):
                                if boundary < len(sequence_rewards):
                                    f.write(f"  Boundary {j} (pos {boundary}): {sequence_rewards[boundary]:.4f}\n")
                            
                            f.write(f"\n=== STATISTICS ===\n")
                            stat_names = ['format_reward', 'acc', 'execution', 'tool_use_count', 'run_correct', 'on_topic', 'good_tests']
                            for j, (name, value) in enumerate(zip(stat_names, stats)):
                                f.write(f"{name}: {value:.4f}\n")
                            
                            f.write(f"\n=== TEST CASE INFO ===\n")
                            f.write(f"Test case: {test_case}\n")
                            f.write(f"Extra info: {extra_info}\n")
                            
                            # Summary
                            total_process_reward = sum(sequence_rewards)
                            
                            # Calculate total masked reward
                            total_masked_reward = 0.0
                            valid_masked_count = 0
                            for j in range(len(sequence_rewards)):
                                if (j < len(loss_mask) and j < len(attention_mask) and 
                                    isinstance(loss_mask[j], (int, float)) and isinstance(attention_mask[j], (int, float))):
                                    masked_reward = sequence_rewards[j] * loss_mask[j] * attention_mask[j]
                                    total_masked_reward += masked_reward
                                    if masked_reward != 0.0:
                                        valid_masked_count += 1
                            
                            f.write(f"\n=== SUMMARY ===\n")
                            f.write(f"Total process reward: {total_process_reward:.4f}\n")
                            f.write(f"Total masked reward: {total_masked_reward:.4f}\n")
                            f.write(f"Final reward: {final_reward:.4f}\n")
                            f.write(f"Total tokens: {len(valid_response_ids)}\n")
                            f.write(f"Total segments: {len(unique_segments)}\n")
                            
                            # Mask statistics
                            if loss_mask:
                                active_loss_count = sum(1 for mask in loss_mask if mask == 1 or mask == 1.0)
                                f.write(f"Active loss mask tokens: {active_loss_count}/{len(loss_mask)}\n")
                            
                            if attention_mask:
                                active_att_count = sum(1 for mask in attention_mask if mask == 1 or mask == 1.0)
                                f.write(f"Active attention mask tokens: {active_att_count}/{len(attention_mask)}\n")
                            
                            # Reward vs mask alignment
                            if loss_mask and sequence_rewards:
                                non_zero_rewards = sum(1 for r in sequence_rewards if r != 0.0)
                                active_loss = sum(1 for mask in loss_mask if mask == 1 or mask == 1.0)
                                f.write(f"Non-zero rewards: {non_zero_rewards}, Active loss mask: {active_loss}\n")
                            
                            # Process evaluation scores for this sample
                            f.write(f"\n=== PROCESS EVALUATION SCORES ===\n")
                            if process_scores_for_sample:
                                f.write(f"Number of process evaluation pairs: {len(process_scores_for_sample)}\n")
                                for j, (rc, ot, gt) in enumerate(process_scores_for_sample):
                                    f.write(f"  Pair {j}: run_correct={rc:.4f}, on_topic={ot:.4f}, good_tests={gt:.4f}\n")
                            else:
                                f.write("No process evaluation scores available for this sample.\n")
                            
                        print(f"Multi-segment sample data saved to: {filename}")
                        return  # Save only the first multi-segment sample found
            
            print("No multi-segment samples found, no data saved.")
            
        except Exception as e:
            print(f"Failed to save multi-segment sample data: {e}")
            import traceback
            traceback.print_exc()