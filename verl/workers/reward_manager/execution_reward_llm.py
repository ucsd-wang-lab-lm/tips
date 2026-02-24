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
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass

import re
import concurrent.futures
import os
import time
import threading
import numpy as np

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


# ============================
# Data structures for execution reward
# ============================
@dataclass
class ExecutionResult:
    segment_rewards: List[float]  # Execution rewards for each segment
    has_errors: List[bool]        # Whether each segment has execution errors
    error_types: List[str]        # Types of errors found in each segment


# ============================
# Boundary detection and segmentation (same as info_reward_llm.py)
# ============================
CLOSE_TAGS = [r"</tool_response>"]
CLOSE_TAG_REGEX = re.compile("|".join(CLOSE_TAGS), re.IGNORECASE)

def _find_tool_response_boundaries_in_text(response_text: str) -> List[int]:
    """Find </tool_response> boundaries (inclusive char index) in response text."""
    bounds = []
    for m in CLOSE_TAG_REGEX.finditer(response_text):
        end_pos = m.end() - 1  # inclusive
        bounds.append(end_pos)
    return bounds


def _build_seg_ids_from_tokens(token_strings: List[str]) -> List[int]:
    """
    Build seg_ids for each token. Each segment runs until a closing tool response tag
    ("</tool_response>"). The final tail segment uses -1.
    If no tool responses are found, all tokens are -1.
    """
    resp_len = len(token_strings)
    if resp_len == 0:
        return []

    closing_tags = ["</tool_response>"]
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


# ============================
# Execution error detection
# ============================
def _extract_tool_segments(response_text: str) -> List[str]:
    """Extract tool response segments from response text."""
    segments = []
    
    # Find all tool_response blocks (consistent with naive_llm.py)
    tool_response_pattern = r'<tool_response>(.*?)</tool_response>'
    matches = re.finditer(tool_response_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        segment_content = match.group(1).strip()
        segments.append(segment_content)
    
    return segments


def _check_tool_execution_and_search_results(
    segment_content: str, 
    full_response: str,
    accepted_answers: List[str] = None
) -> Tuple[bool, str, float]:
    """
    Check for tool execution and search result rewards based on the specified rules.
    
    Rules:
    1. Tool Execution Reward: Awards 0.2 if the tool is correctly executed 
       (properly formatted tool calls and response doesn't begin with "Error:")
    2. Search Result Answer Presence: Awards 0.5 if any accepted answer appears 
       in the search results (case-insensitive comparison)
    
    Returns:
        has_error: Whether there's an execution error
        error_type: Type of error found or success type
        reward: Execution reward
    """
    total_reward = 0.0
    reward_components = []
    
    # Check if response starts with "Error:"
    if segment_content.strip().startswith("Error:"):
        return True, 'error_response', 0.0
    
    # Tool Execution Reward: 0.2 if tool is correctly executed
    # Check if we have properly formatted tool calls in the full response (consistent with naive_llm.py)
    tool_call_pattern = r'<tool_call>.*?</tool_call>'
    if re.search(tool_call_pattern, full_response, re.DOTALL | re.IGNORECASE):
        # Tool was called and response doesn't start with "Error:"
        total_reward += 0.1
        reward_components.append("tool_execution")
    
    # Search Result Answer Presence: 0.5 if accepted answer appears in results
    if accepted_answers:
        # Use the entire tool_response content for search result checking
        # (the segment_content is already extracted from <tool_response>...</tool_response>)
        search_content = segment_content.lower()
        
        # Check if any accepted answer appears in the tool response (case-insensitive)
        for answer in accepted_answers:
            if answer.lower() in search_content:
                total_reward += 0.15
                reward_components.append("answer_found")
                break  # Only award once even if multiple answers found
    
    # Determine the type based on components
    if reward_components:
        reward_type = '_'.join(reward_components)
    elif len(segment_content.strip()) > 0:
        reward_type = 'no_reward_conditions_met'
    else:
        reward_type = 'empty'
    
    return False, reward_type, total_reward


def _compute_execution_rewards_for_sample(
    response_text: str,
    accepted_answers: List[str] = None,
    execution_reward_scale: float = 1.0,
    verbose: bool = False
) -> ExecutionResult:
    """
    Compute execution rewards for a single sample by checking tool response segments.
    
    Args:
        response_text: The response text containing tool responses
        accepted_answers: List of accepted answers to search for in results
        execution_reward_scale: Scale factor for execution rewards
        verbose: Whether to print verbose information
    
    Returns:
        ExecutionResult with segment rewards and error information
    """
    # Extract tool response segments
    segments = _extract_tool_segments(response_text)
    
    if not segments:
        # No tool responses found
        return ExecutionResult(
            segment_rewards=[],
            has_errors=[],
            error_types=[]
        )
    
    segment_rewards = []
    has_errors = []
    error_types = []
    
    for i, segment in enumerate(segments):
        has_error, error_type, base_reward = _check_tool_execution_and_search_results(
            segment, response_text, accepted_answers
        )
        
        # Apply scaling
        scaled_reward = base_reward * execution_reward_scale
        
        segment_rewards.append(scaled_reward)
        has_errors.append(has_error)
        error_types.append(error_type)
        
        if verbose:
            print(f"Segment {i}: {error_type}, reward={scaled_reward:.3f}, has_error={has_error}")
    
    return ExecutionResult(
        segment_rewards=segment_rewards,
        has_errors=has_errors,
        error_types=error_types
    )


def _per_token_execution_rewards_with_mask(
    seg_ids: List[int], 
    segment_rewards: List[float], 
    response_mask: List[int]
) -> List[float]:
    """
    Create a per-token reward vector by assigning execution rewards at the last valid token
    (response_mask=1) in each segment.
    
    Args:
        seg_ids: Segment IDs for each token (computed based on </tool_response> boundaries)
        segment_rewards: Execution rewards for each segment
        response_mask: Response mask indicating valid tokens (1=valid, 0=invalid)
    
    Returns:
        Per-token reward vector with rewards placed at last valid token in each segment
    """
    resp_len = len(seg_ids)
    rewards = [0.0] * resp_len

    # Build non -1 segment boundaries
    boundaries = _compute_segment_boundaries(seg_ids)
    if not boundaries or not segment_rewards:
        return rewards

    num_segments = len(boundaries)
    per_segment_exec = [0.0] * num_segments
    
    # Assign segment_rewards to segments
    for i in range(min(len(segment_rewards), num_segments)):
        per_segment_exec[i] = float(segment_rewards[i])
    
    # If we have more segment_rewards than segments, accumulate extras to the last segment
    if len(segment_rewards) > num_segments:
        per_segment_exec[-1] += sum(float(v) for v in segment_rewards[num_segments:])

    # For each segment, place reward at the last valid token
    for seg_idx, segment_end_idx in enumerate(boundaries):
        seg_reward = per_segment_exec[seg_idx]
        if seg_reward == 0.0:
            continue
            
        # Find the segment start
        if seg_idx == 0:
            segment_start_idx = 0
        else:
            segment_start_idx = boundaries[seg_idx - 1] + 1
        
        # Find the last valid token (response_mask=1) in this segment
        reward_pos = None
        if response_mask and len(response_mask) >= resp_len:
            for pos in range(segment_end_idx, segment_start_idx - 1, -1):
                if 0 <= pos < len(response_mask) and response_mask[pos] == 1:
                    reward_pos = pos
                    break
        
        # Fallback: if no valid token found or no response_mask, use segment end
        if reward_pos is None:
            reward_pos = segment_end_idx
        
        # Place the reward at the chosen position
        if reward_pos is not None and 0 <= reward_pos < resp_len:
            rewards[reward_pos] += seg_reward

    return rewards


def _per_token_execution_rewards_distributed(
    seg_ids: List[int], 
    segment_rewards: List[float], 
    response_mask: List[int]
) -> List[float]:
    """
    Create a per-token reward vector by distributing execution rewards evenly across
    all valid tokens in each segment.
    
    Args:
        seg_ids: Segment IDs for each token (computed based on </tool_response> boundaries)
        segment_rewards: Execution rewards for each segment
        response_mask: Response mask indicating valid tokens (1=valid, 0=invalid)
    
    Returns:
        Per-token reward vector with rewards distributed across all valid tokens in each segment
    """
    resp_len = len(seg_ids)
    rewards = [0.0] * resp_len

    # Build non -1 segment boundaries
    boundaries = _compute_segment_boundaries(seg_ids)
    if not boundaries or not segment_rewards:
        return rewards

    num_segments = len(boundaries)
    per_segment_exec = [0.0] * num_segments
    
    # Assign segment_rewards to segments
    for i in range(min(len(segment_rewards), num_segments)):
        per_segment_exec[i] = float(segment_rewards[i])
    
    # If we have more segment_rewards than segments, accumulate extras to the last segment
    if len(segment_rewards) > num_segments:
        per_segment_exec[-1] += sum(float(v) for v in segment_rewards[num_segments:])

    # For each segment, distribute reward across all valid tokens
    for seg_idx, segment_end_idx in enumerate(boundaries):
        seg_reward = per_segment_exec[seg_idx]
        if seg_reward == 0.0:
            continue
            
        # Find the segment start
        if seg_idx == 0:
            segment_start_idx = 0
        else:
            segment_start_idx = boundaries[seg_idx - 1] + 1
        
        # Find all valid tokens (response_mask=1) in this segment
        valid_positions = []
        if response_mask and len(response_mask) >= resp_len:
            for pos in range(segment_start_idx, segment_end_idx + 1):
                if 0 <= pos < len(response_mask) and response_mask[pos] == 1:
                    valid_positions.append(pos)
        
        # Fallback: if no valid tokens found or no response_mask, use all tokens in segment
        if not valid_positions:
            valid_positions = list(range(segment_start_idx, min(segment_end_idx + 1, resp_len)))
        
        # Distribute reward evenly across valid positions
        if valid_positions:
            reward_per_token = seg_reward
            for pos in valid_positions:
                if 0 <= pos < resp_len:
                    rewards[pos] += reward_per_token

    return rewards


# ============================
# Multi-threaded worker function  
# ============================
def _process_execution_reward_batch(
    batch_tasks: List[Tuple[int, str, List[str], List[str], float, bool]],
    execution_reward_scale: float = 1.0
) -> Dict[int, Tuple[List[int], List[float], dict]]:
    """
    Process a batch of execution reward tasks.
    
    Args:
        batch_tasks: List of (index, response_text, token_strings, accepted_answers, execution_reward_scale, verbose)
        execution_reward_scale: Scale factor for execution rewards
        
    Returns:
        Dict mapping index to (seg_ids, execution_rewards, stats)
    """
    results = {}
    
    if not batch_tasks:
        return results
    
    for task in batch_tasks:
        (index, response_text, token_strings, accepted_answers, exec_scale, verbose) = task
        
        try:
            # Compute execution rewards
            exec_result = _compute_execution_rewards_for_sample(
                response_text=response_text,
                accepted_answers=accepted_answers,
                execution_reward_scale=exec_scale,
                verbose=verbose
            )
            
            # Build segment IDs from tokens
            seg_ids = _build_seg_ids_from_tokens(token_strings)
            
            # Convert segment rewards to per-token rewards (will be adjusted later with mask)
            execution_rewards = [0.0] * len(token_strings)
            
            # Collect stats
            stats = {
                "num_segments": len(exec_result.segment_rewards),
                "total_execution_reward": sum(exec_result.segment_rewards),
                "num_errors": sum(exec_result.has_errors),
                "error_rate": sum(exec_result.has_errors) / max(1, len(exec_result.has_errors)),
            }
            
            # Add detailed execution statistics
            if exec_result.segment_rewards:
                exec_arr = np.array(exec_result.segment_rewards)
                stats.update({
                    "exec_reward_mean": float(np.mean(exec_arr)),
                    "exec_reward_std": float(np.std(exec_arr)),
                    "exec_reward_max": float(np.max(exec_arr)),
                    "exec_reward_min": float(np.min(exec_arr)),
                    "exec_positive_count": int(np.sum(exec_arr > 0)),
                    "exec_negative_count": int(np.sum(exec_arr < 0)),
                    "exec_zero_count": int(np.sum(np.abs(exec_arr) < 1e-6)),
                })
                
                # Error type distribution
                error_type_counts = {}
                for error_type in exec_result.error_types:
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
                stats["error_type_counts"] = error_type_counts
            else:
                stats.update({
                    "exec_reward_mean": 0.0,
                    "exec_reward_std": 0.0,
                    "exec_reward_max": 0.0,
                    "exec_reward_min": 0.0,
                    "exec_positive_count": 0,
                    "exec_negative_count": 0,
                    "exec_zero_count": 0,
                    "error_type_counts": {}
                })
            
            results[index] = (seg_ids, execution_rewards, stats, exec_result.segment_rewards)
            
        except Exception as e:
            if verbose:
                print(f"Error processing execution reward for sample {index}: {e}")
            # Return empty/zero results on error
            seg_ids = [-1] * len(token_strings) if token_strings else []
            execution_rewards = [0.0] * len(token_strings) if token_strings else []
            stats = {
                "num_segments": 0, "total_execution_reward": 0.0, 
                "num_errors": 0, "error_rate": 0.0
            }
            results[index] = (seg_ids, execution_rewards, stats, [])
    
    return results


def _save_sample_data_execution_reward(data, reward_tensor, main_extra_by_index, exec_results_by_index, 
                                     prompt_strs, response_strs, tokenizer, max_samples: int = 2):
    """
    Save detailed sample data for debugging and analysis.
    Focuses on samples with execution rewards and tool use boundaries.
    """
    try:
        # Create output directory if not exists
        output_dir = "reward_samples_execution_reward_llm"
        os.makedirs(output_dir, exist_ok=True)
        
        saved_count = 0
        timestamp = int(time.time())
        
        # Iterate through data to find interesting samples
        for i in range(min(len(data), 20)):  # Check first 20 samples
            if saved_count >= max_samples:
                break
                
            data_item = data[i]
            
            # Check if this sample has execution rewards
            if i not in exec_results_by_index:
                continue
            
            seg_ids, exec_rewards, exec_stats, segment_rewards = exec_results_by_index[i]
            
            # Only save samples with execution rewards (segments found)
            unique_segments = set(seg_id for seg_id in seg_ids if seg_id >= 0)
            if len(unique_segments) >= 1 and exec_stats.get("num_segments", 0) > 0:
                filename = f"{output_dir}/execution_reward_llm_sample_{saved_count}_{timestamp}.txt"
                
                # Get response data
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                response_ids = data_item.batch["responses"]
                valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
                valid_response_ids = response_ids[:valid_response_length]
                
                # Get response_mask from batch if available
                if "response_mask" in data_item.batch:
                    response_mask_tensor = data_item.batch["response_mask"][:valid_response_length]
                    response_mask = response_mask_tensor.cpu().tolist()
                else:
                    # Fallback: compute from attention_mask if response_mask not available
                    full_attention_mask = data_item.batch["attention_mask"]
                    response_attention_mask = full_attention_mask[prompt_length:prompt_length + valid_response_length]
                    response_mask = response_attention_mask.cpu().tolist()
                
                # Get rewards (sequence rewards + final reward)
                sequence_rewards = reward_tensor[i, :valid_response_length].cpu().tolist()
                final_reward = reward_tensor[i, -1].item()
                
                # Get additional info
                ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
                data_source = data_item.non_tensor_batch.get("data_source", "")
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                
                # Get main extra info for this sample
                main_extra = main_extra_by_index.get(i, {})
                
                # Get prompt and response strings
                prompt_str = prompt_strs.get(i, "")
                response_str = response_strs.get(i, "")
                
                # Get individual token strings
                token_strings = []
                for token_id in valid_response_ids:
                    token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                    token_strings.append(token_str)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=== EXECUTION REWARD LLM MANAGER SAMPLE DATA ===\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Sample Index: {i}\n")
                    f.write(f"Unique tool use segments: {sorted(unique_segments)} (Total: {len(unique_segments)})\n")
                    f.write(f"Execution reward segments: {exec_stats.get('num_segments', 0)}\n")
                    f.write(f"Total execution reward: {exec_stats.get('total_execution_reward', 0.0):.6f}\n")
                    f.write(f"Number of errors: {exec_stats.get('num_errors', 0)}\n")
                    f.write(f"Error rate: {exec_stats.get('error_rate', 0.0):.3f}\n\n")
                    
                    f.write("=== PROMPT ===\n")
                    f.write(f"Length: {len(prompt_str)} chars\n")
                    f.write(f"Content: {prompt_str[:500]}{'...' if len(prompt_str) > 500 else ''}\n\n")
                    
                    f.write("=== DECODED RESPONSE ===\n")
                    f.write(f"Length: {len(valid_response_ids)} tokens\n")
                    f.write(f"Decoded: {response_str}\n\n")
                    
                    f.write("=== TOKEN-LEVEL DETAILS ===\n")
                    f.write("Format: idx | token_id | token_string | seg_id | seq_reward | exec_reward | response_mask\n")
                    f.write("-" * 110 + "\n")
                    
                    for j, (token_id, token_str, seq_reward) in enumerate(zip(valid_response_ids, token_strings, sequence_rewards)):
                        seg_id_str = str(seg_ids[j]) if j < len(seg_ids) else "N/A"
                        exec_reward = exec_rewards[j] if j < len(exec_rewards) else 0.0
                        response_mask_val = response_mask[j] if j < len(response_mask) else "N/A"
                        
                        f.write(f"{j:3d}: {token_id:6d} | {repr(token_str):20s} | {seg_id_str:>5s} | {seq_reward:8.4f} | {exec_reward:8.4f} | {response_mask_val:>12}\n")
                    
                    f.write(f"\n=== REWARDS SUMMARY ===\n")
                    non_zero_seq_rewards = [(j, r) for j, r in enumerate(sequence_rewards) if r != 0.0]
                    f.write(f"Non-zero sequence reward positions: {len(non_zero_seq_rewards)}\n")
                    for pos, reward in non_zero_seq_rewards:
                        f.write(f"  Position {pos}: {reward:.4f}\n")
                    
                    non_zero_exec_rewards = [(j, r) for j, r in enumerate(exec_rewards) if r != 0.0]
                    f.write(f"Non-zero execution reward positions: {len(non_zero_exec_rewards)}\n")
                    for pos, reward in non_zero_exec_rewards:
                        f.write(f"  Position {pos}: {reward:.4f}\n")
                    
                    f.write(f"Final reward (T+1): {final_reward:.4f}\n")
                    
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
                    boundaries = _compute_segment_boundaries(seg_ids)
                    f.write(f"Segment boundaries (end positions): {boundaries}\n")
                    
                    # Per-segment execution rewards
                    f.write(f"Per-segment execution rewards:\n")
                    for seg_idx, reward_val in enumerate(segment_rewards):
                        f.write(f"  Segment {seg_idx}: {reward_val:.6f}\n")
                    
                    f.write(f"\n=== METADATA ===\n")
                    f.write(f"Data source: {data_source}\n")
                    f.write(f"Ground truth: {ground_truth}\n")
                    f.write(f"Extra info: {extra_info}\n")
                    
                    f.write(f"\n=== MAIN SCORE EXTRA INFO ===\n")
                    if isinstance(main_extra, dict) and main_extra:
                        for key, value in main_extra.items():
                            f.write(f"{key}: {value}\n")
                    else:
                        f.write("No extra info from main score\n")
                    
                    f.write(f"\n=== EXECUTION REWARD STATS ===\n")
                    for key, value in exec_stats.items():
                        if key == "error_type_counts":
                            f.write(f"{key}: {value}\n")
                        elif "count" in key:
                            f.write(f"{key}: {value}\n")
                        else:
                            f.write(f"{key}: {value:.6f}\n")
                    
                    # Summary statistics
                    total_seq_reward = sum(sequence_rewards)
                    total_exec_reward = sum(exec_rewards)
                    
                    f.write(f"\n=== SUMMARY ===\n")
                    f.write(f"Total sequence reward: {total_seq_reward:.4f}\n")
                    f.write(f"Total execution reward: {total_exec_reward:.4f}\n")
                    f.write(f"Final reward (T+1): {final_reward:.4f}\n")
                    f.write(f"Combined total: {total_seq_reward + final_reward:.4f}\n")
                    f.write(f"Total tokens: {len(valid_response_ids)}\n")
                    f.write(f"Total segments: {len(unique_segments)}\n")
                    
                print(f"Execution reward sample data saved to: {filename}")
                saved_count += 1
        
        if saved_count == 0:
            print("No samples with execution rewards found, no data saved.")
        else:
            print(f"Saved {saved_count} execution reward sample(s) to {output_dir}/")
            
    except Exception as e:
        print(f"Failed to save execution reward sample data: {e}")
        import traceback
        traceback.print_exc()


# ============================
# Main task processing
# ============================
def _process_main_task(
    index: int,
    response_str: str,
    ground_truth: Any,
    data_source: str,
    extra_info: dict,
    compute_score_fn: Any,
    score_source: str
) -> Tuple[int, float, dict, dict]:
    """Process main compute_score task."""
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


@register("execution_reward")
class ExecutionRewardManager(AbstractRewardManager):
    """
    Reward manager that combines main task scoring with execution error checking.
    
    Features:
    - Computes main task rewards using the provided compute_score function
    - Calculates execution rewards by checking for tool call errors
    - Distributes execution rewards at tool-use segment boundaries
    - Places main reward at the final token
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 3,
        compute_score=None,
        eval: bool = False,
        reward_fn_key: str = "data_source",
        num_threads: int = 4,
        enable_execution_reward: bool = True,
        score_source: str = "em",
        execution_reward_weight: float = 1.0,
        execution_reward_scale: float = 1.0,
        reward_distribution_mode: str = "last_token",
        verbose: bool = False
    ) -> None:
        """
        Initialize the ExecutionRewardLLMManager.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging.
            compute_score: A function to compute the main task reward score.
            eval: Whether this is evaluation mode.
            reward_fn_key: The key used to access the data source in the non-tensor batch data.
            num_threads: Number of threads for parallel execution reward computation.
            enable_execution_reward: Whether to enable execution reward computation.
            score_source: Score source for main task computation.
            execution_reward_weight: Weight for execution rewards in the final reward tensor.
            execution_reward_scale: Scale factor for execution rewards.
            reward_distribution_mode: How to distribute segment rewards. Options:
                - "last_token": Place reward at the last valid token of each segment (default)
                - "distributed": Distribute reward evenly across all valid tokens in each segment
            verbose: Whether to enable verbose logging.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_threads = max(1, int(num_threads))
        self.enable_execution_reward = bool(enable_execution_reward)
        self.score_source = score_source
        self.execution_reward_weight = float(execution_reward_weight)
        self.execution_reward_scale = float(execution_reward_scale)
        self.verbose = verbose
        
        # Validate and set reward distribution mode
        valid_modes = ["last_token", "distributed"]
        if reward_distribution_mode not in valid_modes:
            raise ValueError(f"reward_distribution_mode must be one of {valid_modes}, got: {reward_distribution_mode}")
        self.reward_distribution_mode = reward_distribution_mode
        
        if self.enable_execution_reward:
            print(f"ExecutionRewardLLMManager initialized")
            print(f"Using {self.num_threads} threads for execution reward computation")
            print(f"Execution reward weight: {self.execution_reward_weight:.2f}, scale: {self.execution_reward_scale:.2f}")
            print(f"Reward distribution mode: {self.reward_distribution_mode}")
            print(f"Tool Execution Reward: 0.2 for correct tool execution")
            print(f"Search Result Answer Presence: 0.5 if accepted answer found in results")
        else:
            print(f"ExecutionRewardLLMManager initialized with execution rewards disabled")

    def _extract_accepted_answers(self, ground_truth: Any) -> List[str]:
        """
        Extract accepted answers from ground_truth data.
        
        Args:
            ground_truth: Ground truth data, may contain various answer formats
            
        Returns:
            List of accepted answer strings for searching in results
        """
        accepted_answers = []
        
        if not isinstance(ground_truth, dict):
            return accepted_answers
        
        # Try to extract from different possible fields
        possible_fields = ['target']
        
        for field in possible_fields:
            if field in ground_truth:
                answers_data = ground_truth[field]
                
                # Handle numpy array
                try:
                    import numpy as np
                    if isinstance(answers_data, np.ndarray):
                        answers_data = answers_data.tolist()
                except ImportError:
                    pass
                
                # Handle list
                if isinstance(answers_data, list):
                    for item in answers_data:
                        if isinstance(item, str) and item.strip():
                            accepted_answers.append(item.strip())
                        elif item is not None:
                            accepted_answers.append(str(item).strip())
                
                # Handle single string
                elif isinstance(answers_data, str) and answers_data.strip():
                    accepted_answers.append(answers_data.strip())
                
                # Handle other types
                elif answers_data is not None:
                    try:
                        accepted_answers.append(str(answers_data).strip())
                    except Exception:
                        pass
                
                # If we found answers in this field, use them
                if accepted_answers:
                    break
        
        return accepted_answers

    def _print_execution_summary(self, exec_results_by_index: dict) -> None:
        """Print a summary of execution rewards."""
        print("\n=== EXECUTION REWARD SUMMARY ===")
        
        total_segments = 0
        total_errors = 0
        total_positive_rewards = 0
        total_negative_rewards = 0
        exec_rewards_list = []
        
        for sample_idx, (seg_ids, exec_rewards, exec_stats, segment_rewards) in exec_results_by_index.items():
            num_segments = exec_stats.get("num_segments", 0)
            num_errors = exec_stats.get("num_errors", 0)
            total_exec_reward = exec_stats.get("total_execution_reward", 0.0)
            
            total_segments += num_segments
            total_errors += num_errors
            exec_rewards_list.extend(segment_rewards)
            
            print(f"Sample {sample_idx}: {num_segments} segments, {num_errors} errors, total_exec_reward={total_exec_reward:.4f}")
            
            for reward_val in segment_rewards:
                if reward_val > 0:
                    total_positive_rewards += 1
                elif reward_val < 0:
                    total_negative_rewards += 1
        
        # Overall statistics
        if exec_rewards_list:
            avg_reward = np.mean(exec_rewards_list)
            std_reward = np.std(exec_rewards_list)
            max_reward = max(exec_rewards_list)
            min_reward = min(exec_rewards_list)
            
            print(f"\nOverall stats: {total_segments} segments, {total_errors} errors")
            print(f"  Error rate: {100*total_errors/max(1, total_segments):.1f}%")
            print(f"  Positive rewards: {total_positive_rewards} ({100*total_positive_rewards/max(1, total_segments):.1f}%)")
            print(f"  Negative rewards: {total_negative_rewards} ({100*total_negative_rewards/max(1, total_segments):.1f}%)")
            print(f"  Zero rewards: {total_segments-total_positive_rewards-total_negative_rewards}")
            print(f"  Avg reward: {avg_reward:.4f}, Std: {std_reward:.4f}")
            print(f"  Range: [{min_reward:.4f}, {max_reward:.4f}]")
        
        print("=" * 40)

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Process data and compute rewards.
        
        Args:
            data: DataProto containing batch data
            return_dict: Whether to return dict format
        """
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        # Create reward_tensor with shape based on distribution mode
        responses_shape = data.batch['responses'].shape  # (bs, T)
        bs, T = responses_shape
        if self.reward_distribution_mode == "distributed":
            # distributed mode needs extra column for final reward
            reward_tensor = torch.zeros((bs, T + 1), dtype=torch.float32)
        else:
            # last_token mode uses existing response positions
            reward_tensor = torch.zeros((bs, T), dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # Create stats tensor: [format_reward, f1, em, execution_reward_total, num_segments, 
        #                      num_errors, error_rate, exec_reward_mean, exec_reward_std, exec_reward_max, 
        #                      exec_reward_min, exec_positive_count, exec_negative_count, exec_zero_count, total_samples]
        stats_tensor = torch.zeros((bs, 15), dtype=torch.float32)

        already_print_data_sources = {}

        # Pre-process data and prepare for batch decoding
        batch_valid_prompt_ids = []
        batch_valid_response_ids = []
        batch_metadata = []
        
        for i in range(len(data)):
            data_item = data[i]

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
            if hasattr(self.tokenizer, 'batch_decode'):
                prompt_strs_list = self.tokenizer.batch_decode(batch_valid_prompt_ids, skip_special_tokens=True)
                response_strs_list = self.tokenizer.batch_decode(batch_valid_response_ids, skip_special_tokens=False)
            else:
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
            if hasattr(self.tokenizer, 'batch_decode'):
                # Prepare all single-token lists for batch decode
                all_single_tokens = []
                token_counts = []
                for response_ids in batch_valid_response_ids:
                    single_token_lists = [[int(t)] for t in response_ids]
                    all_single_tokens.extend(single_token_lists)
                    token_counts.append(len(response_ids))
                
                # Batch decode all single tokens at once
                all_decoded = self.tokenizer.batch_decode(all_single_tokens, skip_special_tokens=False)
                
                # Reconstruct per-sample token strings
                start_idx = 0
                for count in token_counts:
                    token_strings = all_decoded[start_idx:start_idx + count]
                    batch_token_strings.append(token_strings)
                    start_idx += count
            else:
                # Fallback: try convert_ids_to_tokens if batch_decode not available
                if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                    for response_ids in batch_valid_response_ids:
                        converted = self.tokenizer.convert_ids_to_tokens(response_ids)
                        token_strings = [t if isinstance(t, str) else "" for t in converted]
                        batch_token_strings.append(token_strings)
                else:
                    raise AttributeError("Neither batch_decode nor convert_ids_to_tokens available")
        except Exception as e:
            print(f"Batch token conversion failed, falling back to individual decode: {e}")
            # Final fallback: individual decode
            for response_ids in batch_valid_response_ids:
                token_strings = [self.tokenizer.decode([int(t)], skip_special_tokens=False) for t in response_ids]
                batch_token_strings.append(token_strings)

        # Prepare tasks for parallel processing
        execution_reward_tasks = []
        main_tasks = []
        
        for idx, (prompt_str, response_str, token_strings, metadata) in enumerate(
            zip(prompt_strs_list, response_strs_list, batch_token_strings, batch_metadata)
        ):
            i = metadata['index']
            
            # Main task (always processed)
            main_tasks.append((
                i, response_str, metadata['ground_truth'], metadata['data_source'],
                metadata['extra_info'], self.compute_score, self.score_source
            ))
            
            # Execution reward task (if enabled)
            if self.enable_execution_reward:
                # Extract accepted answers from ground truth
                accepted_answers = self._extract_accepted_answers(metadata['ground_truth'])
                
                execution_reward_tasks.append((
                    i, response_str, token_strings, accepted_answers, self.execution_reward_scale, self.verbose
                ))

        # Process main tasks (single-threaded for simplicity)
        main_score_by_index = {}
        main_extra_by_index = {}
        main_stats_by_index = {}
        
        for task in main_tasks:
            index, main_score, main_extra, stats_dict = _process_main_task(*task)
            main_score_by_index[index] = main_score
            main_extra_by_index[index] = main_extra
            main_stats_by_index[index] = stats_dict

        # Process execution reward tasks (multi-threaded)
        exec_results_by_index = {}
        if self.enable_execution_reward and execution_reward_tasks:
            print(f"Computing execution rewards for {len(execution_reward_tasks)} samples using {self.num_threads} threads...")
            
            # Split tasks into chunks for threading
            chunk_size = max(1, len(execution_reward_tasks) // self.num_threads)
            chunks = [execution_reward_tasks[i:i + chunk_size] for i in range(0, len(execution_reward_tasks), chunk_size)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [
                    executor.submit(_process_execution_reward_batch, chunk, self.execution_reward_scale)
                    for chunk in chunks
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_results = future.result()
                        exec_results_by_index.update(chunk_results)
                    except Exception as e:
                        print(f"Error in execution reward computation: {e}")
            
            # Print execution reward summary
            if exec_results_by_index and self.verbose:
                self._print_execution_summary(exec_results_by_index)

        # Aggregate rewards into tensor
        seg_ids_by_index = {}  # Store seg_ids for return
        
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())

            # Add execution rewards per token to sequence positions
            adjusted_exec_rewards = []  # Initialize for stats calculation
            if i in exec_results_by_index:
                seg_ids, exec_rewards, exec_stats, segment_rewards = exec_results_by_index[i]
                # Store seg_ids for this sample
                seg_ids_by_index[i] = seg_ids
                
                if segment_rewards and valid_response_length > 1:  # Need at least 2 tokens (one for exec reward, one for final)
                    # Get response_mask for this sample
                    if "response_mask" in data_item.batch:
                        response_mask_tensor = data_item.batch["response_mask"][:valid_response_length]
                        response_mask = response_mask_tensor.cpu().tolist()
                    else:
                        # Fallback: compute from attention_mask if response_mask not available
                        full_attention_mask = data_item.batch["attention_mask"]
                        response_attention_mask = full_attention_mask[prompt_length:prompt_length + valid_response_length]
                        response_mask = response_attention_mask.cpu().tolist()
                    
                    # Place rewards based on distribution mode
                    if self.reward_distribution_mode == "last_token":
                        adjusted_exec_rewards = _per_token_execution_rewards_with_mask(
                            seg_ids, segment_rewards, response_mask
                        )
                    elif self.reward_distribution_mode == "distributed":
                        adjusted_exec_rewards = _per_token_execution_rewards_distributed(
                            seg_ids, segment_rewards, response_mask
                        )
                    else:
                        # Fallback to last_token mode
                        adjusted_exec_rewards = _per_token_execution_rewards_with_mask(
                            seg_ids, segment_rewards, response_mask
                        )
                    
                    # Ensure we don't exceed tensor bounds
                    if self.reward_distribution_mode == "distributed":
                        max_pos = min(len(adjusted_exec_rewards), valid_response_length, T)  # Don't overwrite final column T+1
                    else:
                        max_pos = min(len(adjusted_exec_rewards), valid_response_length)  # Can use all positions in last_token mode
                    for pos in range(max_pos):
                        reward_tensor[i, pos] += float(adjusted_exec_rewards[pos]) * self.execution_reward_weight
                
                # Update stats tensor with execution reward stats
                stats_tensor[i, 4] = float(exec_stats.get("num_segments", 0))
                stats_tensor[i, 5] = float(exec_stats.get("num_errors", 0))
                stats_tensor[i, 6] = float(exec_stats.get("error_rate", 0.0))
                stats_tensor[i, 7] = float(exec_stats.get("exec_reward_mean", 0.0))
                stats_tensor[i, 8] = float(exec_stats.get("exec_reward_std", 0.0))
                stats_tensor[i, 9] = float(exec_stats.get("exec_reward_max", 0.0))
                stats_tensor[i, 10] = float(exec_stats.get("exec_reward_min", 0.0))
                stats_tensor[i, 11] = float(exec_stats.get("exec_positive_count", 0))
                stats_tensor[i, 12] = float(exec_stats.get("exec_negative_count", 0))
                stats_tensor[i, 13] = float(exec_stats.get("exec_zero_count", 0))
                stats_tensor[i, 14] = 1.0  # total_samples counter
            else:
                # For samples without execution rewards, still generate seg_ids if needed
                # Get token strings for this sample from batch_token_strings if available
                try:
                    # Find the corresponding token_strings from the batch processing
                    for idx, metadata in enumerate(batch_metadata):
                        if metadata['index'] == i:
                            token_strings = batch_token_strings[idx]
                            seg_ids = _build_seg_ids_from_tokens(token_strings)
                            seg_ids_by_index[i] = seg_ids
                            break
                    else:
                        # Fallback: create empty seg_ids if no token_strings found
                        seg_ids_by_index[i] = [-1] * valid_response_length
                except Exception:
                    # Fallback: create empty seg_ids
                    seg_ids_by_index[i] = [-1] * valid_response_length

            # Main score goes to the final valid position
            main_score = main_score_by_index.get(i, 0.0)
            main_extra = main_extra_by_index.get(i, {})
            main_stats = main_stats_by_index.get(i, {})
            
            # Final reward placement based on distribution mode
            if self.reward_distribution_mode == "last_token":
                # Place final reward at the last valid token within the response
                if valid_response_length > 0:
                    reward_tensor[i, valid_response_length - 1] += float(main_score)
            elif self.reward_distribution_mode == "distributed":
                # Place final reward in the extra column (T+1)
                reward_tensor[i, -1] = float(main_score)

            # Store extra info from main compute_score
            if isinstance(main_extra, dict):
                for key, value in main_extra.items():
                    reward_extra_info[key].append(value)
            else:
                reward_extra_info["score"].append(main_score)

            # Populate main stats in stats tensor
            stats_tensor[i, 0] = float(main_stats.get("format_reward", 0.0))
            stats_tensor[i, 1] = float(main_stats.get("f1", 0.0))
            stats_tensor[i, 2] = float(main_stats.get("em", 0.0))
            stats_tensor[i, 3] = float(sum(adjusted_exec_rewards) if adjusted_exec_rewards else 0.0)
            
            # Set total_samples counter for all samples (including those without execution rewards)
            stats_tensor[i, 14] = 1.0

        # Save sample data for debugging and analysis
        if True:
            try:
                # Create mapping from sample index to strings
                prompt_strs_dict = {}
                response_strs_dict = {}
                for idx, metadata in enumerate(batch_metadata):
                    i = metadata['index']
                    prompt_strs_dict[i] = prompt_strs_list[idx]
                    response_strs_dict[i] = response_strs_list[idx]
                
                _save_sample_data_execution_reward(
                    data, reward_tensor, main_extra_by_index, exec_results_by_index, 
                    prompt_strs_dict, response_strs_dict, self.tokenizer, max_samples=1
                )
            except Exception as e:
                print(f"Warning: Failed to save sample data: {e}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "exec_stats_tensor": stats_tensor,
                "exec_results_by_index": exec_results_by_index,
                "seg_ids_by_index": seg_ids_by_index,
                "exec_flag": True,
            }
        else:
            # Return both reward and stats tensors (like naive_llm.py)
            return reward_tensor, stats_tensor
