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
    
    # ======= Replacement start (only replace the given section) =======
import numpy as np  # Recommended to put at file top, but here is also fine

# Add OpenAI import for vLLM information reward evaluation
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("Warning: OpenAI package not found. Information reward evaluation will be disabled.")

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


# ============================
# Data structures for information reward
# ============================
@dataclass
class BoundaryScore:
    i: int                      # boundary index (i=0 is the initial segment)
    phi: float                  # Φ(s_end_i) after candidate aggregation
    per_cand_logp: List[float]  # per-candidate (length-normalized) logP

@dataclass
class InfoRewardResult:
    boundaries: List[BoundaryScore]
    delta_phi: List[float]      # ΔΦ_i = Φ_i - Φ_{i-1}, length M (excluding i=0)


# ============================
# Boundary detection and segmentation
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
    
    This is adapted from naive_llm.py to work with information reward boundaries.
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
# vLLM information reward calculation
# ============================
def _build_prompt_and_span(ctx_text: str, answer_text: str) -> Tuple[str, Tuple[int, int]]:
    """
    Build: prompt = ctx_text + "<answer>" + answer_text + "</answer>"
    Return (prompt, (ans_start_char, ans_end_char)) so we score the entire answer section,
    including the <answer> and </answer> tags.
    """
    open_tag  = "<answer>"
    close_tag = "</answer>"
    prompt = ctx_text + open_tag + answer_text + close_tag
    ans_start = len(ctx_text)  # Start from the beginning of <answer> tag
    ans_end   = len(prompt)    # End at the end of </answer> tag
    return prompt, (ans_start, ans_end)

def _extract_prompt_token_logprobs(resp) -> Dict[str, Optional[List]]:
    """
    Normalize vLLM return structures to:
      {
        "tokens": List[str],
        "token_logprobs": List[float],
        "text_offset": List[int] or None,
        "text": echoed prompt text or None
      }
    """
    ch0 = resp.choices[0]
    lp  = getattr(ch0, "logprobs", None)

    # Common structure (attributes)
    if lp and hasattr(lp, "tokens") and hasattr(lp, "token_logprobs"):
        text_offset = getattr(lp, "text_offset", None)
        echoed_text = getattr(ch0, "text", None)  # prompt echo (often equals the input prompt)
        return {
            "tokens": lp.tokens,
            "token_logprobs": lp.token_logprobs,
            "text_offset": text_offset,
            "text": echoed_text
        }

    # Some clients deserialize as dict
    if isinstance(lp, dict) and "tokens" in lp and "token_logprobs" in lp:
        return {
            "tokens": lp["tokens"],
            "token_logprobs": lp["token_logprobs"],
            "text_offset": lp.get("text_offset", None),
            "text": getattr(ch0, "text", None)
        }

    # Less common: a "prompt" block (adapt here if you see it)
    if isinstance(lp, dict) and "prompt" in lp:
        p = lp["prompt"]
        tokens = [t.get("token") for t in p]
        token_logprobs = [t.get("logprob") for t in p]
        text_offset = [t.get("text_offset") for t in p] if all("text_offset" in t for t in p) else None
        return {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "text_offset": text_offset,
            "text": getattr(ch0, "text", None)
        }

    raise RuntimeError(
        "Could not parse prompt-side tokens/logprobs from response. "
        "Print the response object and adjust `_extract_prompt_token_logprobs` accordingly."
    )


def _reconstruct_offsets_from_echo(tokens: List[str], echoed_text: str) -> List[Optional[int]]:
    """
    Fallback if vLLM does not return text_offset: reconstruct per-token start indices
    by aligning token strings to the echoed prompt text.
    """
    offsets: List[Optional[int]] = []
    cursor = 0
    N = len(echoed_text)

    for tok in tokens:
        if tok == "":
            offsets.append(cursor)
            continue
        # Find the next occurrence of tok at or after cursor
        pos = echoed_text.find(tok, cursor)
        if pos == -1:
            # Couldn't find perfect alignment; mark None and keep cursor unchanged.
            offsets.append(None)
        else:
            offsets.append(pos)
            cursor = pos + len(tok)
            if cursor > N:
                cursor = N
    return offsets


def _score_answer_with_vllm(
    ctx_text: str,
    answer_text: str,
    vllm_client: OpenAI,
    model: str,
    length_norm: bool = False,
    timeout: float = 30.0,
    verbose: bool = False,
) -> float:
    """
    Score (ctx + <answer>answer</answer>) on the prompt side, summing/averaging the
    logprobs ONLY over the answer body. No special tokens required.
    """
    prompt, (ans_start, ans_end) = _build_prompt_and_span(ctx_text, answer_text)

    try:
        resp = vllm_client.completions.create(
            model="qwen2.5-7b-instruct",
            prompt=prompt,
            max_tokens=0,
            temperature=1.0,
            top_p=1.0,
            logprobs=0,
            echo=True,
            timeout=timeout,
            extra_body={
                "prompt_logprobs": 1,      # return logprobs for prompt tokens
                "top_k": -1,
                "repetition_penalty": 1.0,
            }
        )
    except Exception as e:
        if verbose:
            print(f"vLLM API call failed: {e}")
        return 0.0  # Return 0 on API failure

    try:
        plp = _extract_prompt_token_logprobs(resp)
        tokens         = plp["tokens"]
        token_logprobs = plp["token_logprobs"]
        text_offsets   = plp["text_offset"]   # may be None
        echoed_text    = plp.get("text", "")  # with echo=True, this should be the *prompt* text

        # Primary path: use vLLM-provided text offsets if present.
        selected = []
        if text_offsets and any(off is not None for off in text_offsets):
            for lp, off in zip(token_logprobs, text_offsets):
                if off is None:
                    continue
                if ans_start <= off < ans_end and lp is not None:
                    selected.append(lp)

        # Fallback: reconstruct offsets from echoed prompt text.
        if not selected and echoed_text:
            recon = _reconstruct_offsets_from_echo(tokens, echoed_text)
            for lp, off in zip(token_logprobs, recon):
                if off is None:
                    continue
                if ans_start <= off < ans_end and lp is not None:
                    selected.append(lp)

        if not selected:
            if verbose:
                print(f"No tokens selected within answer span for: {answer_text[:50]}...")
            return 0.0

        sum_logp = float(sum(selected))
        avg_logp = sum_logp / max(1, len(selected)) if length_norm else sum_logp
        return avg_logp

    except Exception as e:
        if verbose:
            print(f"Error processing vLLM response: {e}")
        return 0.0


def _compute_info_reward_for_sample(
    prompt_text: str,
    response_text: str,
    answer_candidates: List[str],
    vllm_client: OpenAI,
    model: str,
    length_norm: bool = False,
    timeout: float = 30.0,
    verbose: bool = False,
    ppl_threshold: float = 200.0,
    delta_phi_scale: float = 0.02,
) -> InfoRewardResult:
    """
    Compute information reward (Φ and ΔΦ) for a single sample.
    
    For each boundary i = 0..M:
      - ctx_0 = prompt_text
      - ctx_i = prompt_text + response_text[:end_i+1] for i>=1
      - For each ctx_i, score all candidates and take max -> Φ_i
    Then ΔΦ_i = Φ_i - Φ_{i-1} for i >= 1.
    
    Args:
        ppl_threshold: If initial average log-prob is more negative than log(1/ppl_threshold), 
                      set all information rewards to 0 for the entire sample.
        delta_phi_scale: Linear scaling factor for ΔΦ to achieve [-0.5, 0.5] range (PBRS-preserving).
    """
    # Find boundaries in response text
    boundaries_inclusive = _find_tool_response_boundaries_in_text(response_text)
    
    if not boundaries_inclusive:
        # No tool use boundaries found, return zero rewards
        return InfoRewardResult(
            boundaries=[BoundaryScore(i=0, phi=0.0, per_cand_logp=[0.0] * len(answer_candidates))],
            delta_phi=[]
        )
    
    # Build contexts: initial + after each boundary
    contexts = [prompt_text]
    for end_pos in boundaries_inclusive:
        ctx_i = prompt_text + response_text[: end_pos + 1]
        contexts.append(ctx_i)

    # Check initial perplexity threshold on prompt-only context
    # If initial ppl is too high, return zero rewards for entire sample
    initial_ctx = contexts[0]  # prompt_text
    initial_scores = []
    for answer in answer_candidates:
        score = _score_answer_with_vllm(
            initial_ctx, answer, vllm_client, model,
            length_norm=True,
            timeout=timeout, verbose=verbose
        )
        initial_scores.append(score)
    
    # Check if initial score indicates poor quality (corrected logic)
    max_initial_score = max(initial_scores) if initial_scores else float('-inf')
    # Convert threshold to log-prob: if ppl_threshold=100, then log_prob_threshold = log(1/100) = -4.6
    log_prob_threshold = np.log(1.0 / ppl_threshold) if ppl_threshold > 0 else float('-inf')
    
    if max_initial_score < log_prob_threshold:
        if verbose:
            print(f"Initial max log-prob {max_initial_score:.4f} below threshold {log_prob_threshold:.4f} (ppl > {ppl_threshold}), setting all info rewards to 0")
        # Return zero rewards for entire sample
        zero_boundaries = []
        for i in range(len(contexts)):
            zero_boundaries.append(BoundaryScore(i=i, phi=0.0, per_cand_logp=[0.0] * len(answer_candidates)))
        return InfoRewardResult(boundaries=zero_boundaries, delta_phi=[0.0] * (len(contexts) - 1))

    boundary_scores: List[BoundaryScore] = []

    # Score each context against all answer candidates
    for i, ctx in enumerate(contexts):
        if i == 0:
            # Reuse initial scores to avoid duplicate computation
            per_cand = initial_scores.copy()
        else:
            per_cand = []
            for answer in answer_candidates:
                # Key: Use "joint log probability (sum logp)" instead of average logp
                score = _score_answer_with_vllm(
                    ctx, answer, vllm_client, model,
                    length_norm=False,           # Force disable length normalization
                    timeout=timeout, verbose=verbose
                )
                per_cand.append(score)

        # Aggregate multiple candidate scores:
        # Φ_i = log( sum_k P(a_k | ctx_i) ) = logsumexp( log P(a_k | ctx_i) )
        if per_cand:
            arr = np.asarray(per_cand, dtype=np.float64)

            # Numerically stable logsumexp: m + log(sum(exp(arr - m)))
            m = np.max(arr)
            if np.isfinite(m):
                phi_i = float(m + np.log(np.sum(np.exp(arr - m))))
            else:
                # In extreme cases (e.g., all -inf or NaN), fallback to 0.0 to avoid delta explosion
                phi_i = 0.0
        else:
            # Fallback to 0.0 when no candidates (should not happen in normal flow: ensure candidates or use default fallback)
            phi_i = 0.0

        boundary_scores.append(BoundaryScore(i=i, phi=phi_i, per_cand_logp=per_cand))

    # ========= 单调包络 ΔΦ（避免负奖励）开始 =========
    # 1) 取出各边界的 Φ_i
    phis = np.asarray([b.phi for b in boundary_scores], dtype=np.float64)

    # 2) 对 Φ 做“历史最大值”累积，得到单调包络 F_i = max_{j<=i} Φ_j
    cummax = np.maximum.accumulate(phis)

    # 3) 计算增量 ΔF_i = F_i - F_{i-1}（天然 >= 0）
    delta_arr = cummax[1:] - cummax[:-1] if cummax.size >= 2 else np.asarray([], dtype=np.float64)

    # 4) 线性缩放到目标幅度；线性缩放不破坏 PBRS
    if delta_arr.size:
        delta = (delta_phi_scale * delta_arr).tolist()
    else:
        delta = []
    # ========= 单调包络 ΔΦ（避免负奖励）结束 =========


    return InfoRewardResult(boundaries=boundary_scores, delta_phi=delta)
    # ======= Replacement end =======





def _per_token_info_rewards_simple(seg_ids: List[int], delta_phi: List[float]) -> List[float]:
    """
    Create a per-token reward vector by assigning delta_phi values at segment boundaries.
    This is a simple version used in multi-threading. The main thread will adjust positions later.
    
    Args:
        seg_ids: Segment IDs for each token (computed based on </tool_response> boundaries)
        delta_phi: Information reward deltas for each segment
    
    Returns:
        Per-token reward vector with rewards placed at segment boundaries
    """
    resp_len = len(seg_ids)
    rewards = [0.0] * resp_len

    # Build non -1 segment boundaries (these are still based on </tool_response>)
    boundaries = _compute_segment_boundaries(seg_ids)
    if not boundaries or not delta_phi:
        return rewards

    num_segments = len(boundaries)
    per_segment_info = [0.0] * num_segments
    
    # Assign delta_phi values to segments
    for i in range(min(len(delta_phi), num_segments)):
        per_segment_info[i] = float(delta_phi[i])
    
    # If we have more delta_phi values than segments, accumulate extras to the last segment
    if len(delta_phi) > num_segments:
        per_segment_info[-1] += sum(float(v) for v in delta_phi[num_segments:])

    # Place rewards at segment boundaries (will be adjusted later)
    for seg_idx, segment_end_idx in enumerate(boundaries):
        seg_reward = per_segment_info[seg_idx]
        if seg_reward != 0.0 and 0 <= segment_end_idx < resp_len:
            rewards[segment_end_idx] += seg_reward

    return rewards


def _per_token_info_rewards_with_mask(seg_ids: List[int], delta_phi: List[float], response_mask: List[int]) -> List[float]:
    """
    Create a per-token reward vector by assigning delta_phi values at the last valid token
    (response_mask=1) in each segment.
    
    Strategy: For each segment, place the entire segment reward at the last valid token.
    
    Args:
        seg_ids: Segment IDs for each token (computed based on </tool_response> boundaries)
        delta_phi: Information reward deltas for each segment
        response_mask: Response mask indicating valid tokens (1=valid, 0=invalid)
    
    Returns:
        Per-token reward vector with rewards placed at last valid token in each segment
    """
    resp_len = len(seg_ids)
    rewards = [0.0] * resp_len

    # Build non -1 segment boundaries (these are still based on </tool_response>)
    boundaries = _compute_segment_boundaries(seg_ids)
    if not boundaries or not delta_phi:
        return rewards

    num_segments = len(boundaries)
    per_segment_info = [0.0] * num_segments
    
    # Assign delta_phi values to segments
    for i in range(min(len(delta_phi), num_segments)):
        per_segment_info[i] = float(delta_phi[i])
    
    # If we have more delta_phi values than segments, accumulate extras to the last segment
    if len(delta_phi) > num_segments:
        per_segment_info[-1] += sum(float(v) for v in delta_phi[num_segments:])

    # For each segment, place reward at the last valid token
    for seg_idx, segment_end_idx in enumerate(boundaries):
        seg_reward = per_segment_info[seg_idx]
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




# ============================
# Multi-threaded worker function  
# ============================
def _process_info_reward_batch(
    batch_tasks: List[Tuple[int, str, str, List[str], List[str], str, float, bool]],
    vllm_api_base: str,
    timeout: float = 30.0,
    ppl_threshold: float = 100.0,
    delta_phi_scale: float = 0.02
) -> Dict[int, Tuple[List[int], List[float], dict]]:
    """
    Process a batch of information reward tasks in a single thread.
    
    Args:
        batch_tasks: List of (index, prompt_text, response_text, answer_candidates, token_strings, model, timeout, verbose)
        vllm_api_base: vLLM API base URL
        timeout: Request timeout
        ppl_threshold: Initial perplexity threshold above which all info rewards are set to 0
        delta_phi_scale: Linear scaling factor for ΔΦ to achieve [-0.5, 0.5] range
        
    Returns:
        Dict mapping index to (seg_ids, info_rewards, stats)
    """
    results = {}
    
    if not batch_tasks or not vllm_api_base:
        return results
    
    # Initialize vLLM client for this thread
    vllm_client = None
    try:
        import httpx
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=50,
                keepalive_expiry=30.0
            ),
            timeout=httpx.Timeout(timeout),
        )
        vllm_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            base_url="http://10.24.1.204:9000/v1",
            http_client=http_client,
        )
    except Exception as e:
        print(f"Failed to initialize vLLM client: {e}")
        return results
    
    for task in batch_tasks:
        (index, prompt_text, response_text, answer_candidates, 
         token_strings, model, timeout, verbose) = task
        
        try:
            # Compute information reward
            info_result = _compute_info_reward_for_sample(
                prompt_text=prompt_text,
                response_text=response_text,
                answer_candidates=answer_candidates,
                vllm_client=vllm_client,
                model=model,
                length_norm=True,
                timeout=timeout,
                verbose=verbose,
                ppl_threshold=ppl_threshold,
                delta_phi_scale=delta_phi_scale
            )
            
            # Build segment IDs from tokens
            seg_ids = _build_seg_ids_from_tokens(token_strings)
            
            # Convert delta_phi to per-token rewards at segment boundaries (will be adjusted later)
            info_rewards = _per_token_info_rewards_simple(seg_ids, info_result.delta_phi)
            
            # Collect stats
            stats = {
                "num_boundaries": len(info_result.boundaries),
                "total_delta_phi": sum(info_result.delta_phi),
                "max_phi": max([b.phi for b in info_result.boundaries]) if info_result.boundaries else 0.0,
                "min_phi": min([b.phi for b in info_result.boundaries]) if info_result.boundaries else 0.0,
            }
            
            # Add delta_phi scale statistics
            if info_result.delta_phi:
                import numpy as np
                delta_arr = np.array(info_result.delta_phi)
                # Calculate quantiles with wider coverage (5th, 25th, 50th, 75th, 95th percentiles)
                quantiles = np.percentile(delta_arr, [5, 25, 50, 75, 95])
                stats.update({
                    "delta_phi_mean": float(np.mean(delta_arr)),
                    "delta_phi_std": float(np.std(delta_arr)),
                    "delta_phi_max": float(np.max(delta_arr)),
                    "delta_phi_min": float(np.min(delta_arr)),
                    "delta_phi_abs_mean": float(np.mean(np.abs(delta_arr))),
                    "delta_phi_positive_count": int(np.sum(delta_arr > 0)),
                    "delta_phi_negative_count": int(np.sum(delta_arr < 0)),
                    "delta_phi_zero_count": int(np.sum(np.abs(delta_arr) < 1e-6)),
                    "delta_phi_q05": float(quantiles[0]),      # 5th percentile (extreme low)
                    "delta_phi_q25": float(quantiles[1]),      # 25th percentile
                    "delta_phi_median": float(quantiles[2]),   # 50th percentile (median)
                    "delta_phi_q75": float(quantiles[3]),      # 75th percentile
                    "delta_phi_q95": float(quantiles[4]),      # 95th percentile (extreme high)
                })
            else:
                stats.update({
                    "delta_phi_mean": 0.0,
                    "delta_phi_std": 0.0,
                    "delta_phi_max": 0.0,
                    "delta_phi_min": 0.0,
                    "delta_phi_abs_mean": 0.0,
                    "delta_phi_positive_count": 0,
                    "delta_phi_negative_count": 0,
                    "delta_phi_zero_count": 0,
                    "delta_phi_q05": 0.0,
                    "delta_phi_q25": 0.0,
                    "delta_phi_median": 0.0,
                    "delta_phi_q75": 0.0,
                    "delta_phi_q95": 0.0,
                })
            
            results[index] = (seg_ids, info_rewards, stats, info_result.delta_phi)
            
        except Exception as e:
            if verbose:
                print(f"Error processing info reward for sample {index}: {e}")
            # Return empty/zero results on error
            seg_ids = [-1] * len(token_strings) if token_strings else []
            info_rewards = [0.0] * len(token_strings) if token_strings else []
            stats = {"num_boundaries": 0, "total_delta_phi": 0.0, "max_phi": 0.0, "min_phi": 0.0}
            results[index] = (seg_ids, info_rewards, stats, [])
    
    # Cleanup
    try:
        if vllm_client and hasattr(vllm_client, '_client') and hasattr(vllm_client._client, 'close'):
            vllm_client._client.close()
    except Exception:
        pass
    
    return results


def _save_sample_data_info_reward(data, reward_tensor, main_extra_by_index, info_results_by_index, 
                                 prompt_strs, response_strs, tokenizer, max_samples: int = 2):
    """
    Save detailed sample data for debugging and analysis (adapted from naive_llm.py).
    Focuses on samples with information rewards and tool use boundaries.
    """
    try:
        # Create output directory if not exists
        output_dir = "reward_samples_info_reward_llm"
        os.makedirs(output_dir, exist_ok=True)
        
        saved_count = 0
        timestamp = int(time.time())
        
        # Iterate through data to find interesting samples
        for i in range(min(len(data), 20)):  # Check first 20 samples
            if saved_count >= max_samples:
                break
                
            data_item = data[i]
            
            # Check if this sample has information rewards
            if i not in info_results_by_index:
                continue
            
            seg_ids, info_rewards, info_stats, delta_phi = info_results_by_index[i]
            
            # Only save samples with information rewards (boundaries found)
            unique_segments = set(seg_id for seg_id in seg_ids if seg_id >= 0)  # Ignore -1 padding
            if len(unique_segments) >= 1 and info_stats.get("num_boundaries", 0) > 0:
                filename = f"{output_dir}/info_reward_llm_sample_{saved_count}_{timestamp}.txt"
                
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
                    f.write("=== INFO REWARD LLM MANAGER SAMPLE DATA ===\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Sample Index: {i}\n")
                    f.write(f"Unique tool use segments: {sorted(unique_segments)} (Total: {len(unique_segments)})\n")
                    f.write(f"Info reward boundaries: {info_stats.get('num_boundaries', 0)}\n")
                    f.write(f"Total ΔΦ: {info_stats.get('total_delta_phi', 0.0):.6f}\n")
                    f.write(f"Max Φ: {info_stats.get('max_phi', 0.0):.6f}\n")
                    f.write(f"Min Φ: {info_stats.get('min_phi', 0.0):.6f}\n\n")
                    
                    f.write("=== PROMPT ===\n")
                    f.write(f"Length: {len(prompt_str)} chars\n")
                    f.write(f"Content: {prompt_str[:500]}{'...' if len(prompt_str) > 500 else ''}\n\n")
                    
                    f.write("=== DECODED RESPONSE ===\n")
                    f.write(f"Length: {len(valid_response_ids)} tokens\n")
                    f.write(f"Decoded: {response_str}\n\n")
                    
                    f.write("=== TOKEN-LEVEL DETAILS ===\n")
                    f.write("Format: idx | token_id | token_string | seg_id | seq_reward | info_reward | response_mask\n")
                    f.write("-" * 110 + "\n")
                    
                    for j, (token_id, token_str, seq_reward) in enumerate(zip(valid_response_ids, token_strings, sequence_rewards)):
                        seg_id_str = str(seg_ids[j]) if j < len(seg_ids) else "N/A"
                        info_reward = info_rewards[j] if j < len(info_rewards) else 0.0
                        response_mask_val = response_mask[j] if j < len(response_mask) else "N/A"
                        
                        f.write(f"{j:3d}: {token_id:6d} | {repr(token_str):20s} | {seg_id_str:>5s} | {seq_reward:8.4f} | {info_reward:8.4f} | {response_mask_val:>12}\n")
                    
                    f.write(f"\n=== REWARDS SUMMARY ===\n")
                    non_zero_seq_rewards = [(j, r) for j, r in enumerate(sequence_rewards) if r != 0.0]
                    f.write(f"Non-zero sequence reward positions: {len(non_zero_seq_rewards)}\n")
                    for pos, reward in non_zero_seq_rewards:
                        f.write(f"  Position {pos}: {reward:.4f}\n")
                    
                    non_zero_info_rewards = [(j, r) for j, r in enumerate(info_rewards) if r != 0.0]
                    f.write(f"Non-zero info reward positions: {len(non_zero_info_rewards)}\n")
                    for pos, reward in non_zero_info_rewards:
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
                    
                    # Per-segment info rewards
                    f.write(f"Per-segment info rewards:\n")
                    for seg_idx, end_idx in enumerate(boundaries):
                        if end_idx < len(info_rewards):
                            reward_val = info_rewards[end_idx]
                            f.write(f"  Segment {seg_idx} (ends at {end_idx}): {reward_val:.6f}\n")
                        else:
                            f.write(f"  Segment {seg_idx} (ends at {end_idx}): 0.0 (out of bounds)\n")
                    
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
                    
                    f.write(f"\n=== INFO REWARD STATS ===\n")
                    for key, value in info_stats.items():
                        if key.startswith("delta_phi"):
                            if "count" in key:
                                f.write(f"{key}: {value}\n")
                            else:
                                f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    
                    # Enhanced quantile analysis with 95% coverage
                    if "delta_phi_q05" in info_stats and "delta_phi_q95" in info_stats:
                        iqr = info_stats.get("delta_phi_q75", 0.0) - info_stats.get("delta_phi_q25", 0.0)
                        coverage_95 = info_stats.get("delta_phi_q95", 0.0) - info_stats.get("delta_phi_q05", 0.0)
                        f.write(f"\nDelta Phi Quantile Analysis (95% Coverage):\n")
                        f.write(f"  95% Coverage Range: [{info_stats.get('delta_phi_q05', 0.0):.6f}, {info_stats.get('delta_phi_q95', 0.0):.6f}] (width: {coverage_95:.6f})\n")
                        f.write(f"  IQR (Q75-Q25): {iqr:.6f}\n")
                        f.write(f"  Quartile ranges:\n")
                        f.write(f"    Q05-Q25: [{info_stats.get('delta_phi_q05', 0.0):.6f}, {info_stats.get('delta_phi_q25', 0.0):.6f}]\n")
                        f.write(f"    Q25-Q75: [{info_stats.get('delta_phi_q25', 0.0):.6f}, {info_stats.get('delta_phi_q75', 0.0):.6f}] (50% central)\n")
                        f.write(f"    Q75-Q95: [{info_stats.get('delta_phi_q75', 0.0):.6f}, {info_stats.get('delta_phi_q95', 0.0):.6f}]\n")
                        if info_stats.get("delta_phi_std", 0.0) > 0:
                            mean_val = info_stats.get("delta_phi_mean", 0.0)
                            median_val = info_stats.get("delta_phi_median", 0.0)
                            skewness_indicator = (mean_val - median_val) / info_stats.get("delta_phi_std", 1.0)
                            f.write(f"  Skewness indicator (mean-median)/std: {skewness_indicator:.6f}\n")
                    
                    # Additional delta_phi distribution analysis
                    if "delta_phi_positive_count" in info_stats and "delta_phi_negative_count" in info_stats:
                        total_deltas = info_stats.get("delta_phi_positive_count", 0) + info_stats.get("delta_phi_negative_count", 0) + info_stats.get("delta_phi_zero_count", 0)
                        if total_deltas > 0:
                            pos_pct = 100.0 * info_stats.get("delta_phi_positive_count", 0) / total_deltas
                            neg_pct = 100.0 * info_stats.get("delta_phi_negative_count", 0) / total_deltas
                            zero_pct = 100.0 * info_stats.get("delta_phi_zero_count", 0) / total_deltas
                            f.write(f"\nDelta Phi Distribution:\n")
                            f.write(f"  Positive: {pos_pct:.1f}% ({info_stats.get('delta_phi_positive_count', 0)}/{total_deltas})\n")
                            f.write(f"  Negative: {neg_pct:.1f}% ({info_stats.get('delta_phi_negative_count', 0)}/{total_deltas})\n")
                            f.write(f"  Zero: {zero_pct:.1f}% ({info_stats.get('delta_phi_zero_count', 0)}/{total_deltas})\n")
                    
                    # Summary statistics
                    total_seq_reward = sum(sequence_rewards)
                    total_info_reward = sum(info_rewards)
                    
                    f.write(f"\n=== SUMMARY ===\n")
                    f.write(f"Total sequence reward: {total_seq_reward:.4f}\n")
                    f.write(f"Total info reward: {total_info_reward:.4f}\n")
                    f.write(f"Final reward (T+1): {final_reward:.4f}\n")
                    f.write(f"Combined total: {total_seq_reward + final_reward:.4f}\n")
                    f.write(f"Total tokens: {len(valid_response_ids)}\n")
                    f.write(f"Total segments: {len(unique_segments)}\n")
                    
                print(f"Info reward sample data saved to: {filename}")
                saved_count += 1
        
        if saved_count == 0:
            print("No samples with information rewards found, no data saved.")
        else:
            print(f"Saved {saved_count} info reward sample(s) to {output_dir}/")
            
    except Exception as e:
        print(f"Failed to save info reward sample data: {e}")
        import traceback
        traceback.print_exc()


# ============================
# Main task processing (adapted from naive_llm.py)
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


@register("info_reward_llm_hmax")
class InfoRewardLLMManagerHMax(AbstractRewardManager):
    """
    Reward manager that combines main task scoring with vLLM-based information reward.
    
    Features:
    - Computes main task rewards using the provided compute_score function
    - Calculates information rewards (Φ and ΔΦ) using vLLM API calls
    - Distributes information rewards at tool-use boundaries
    - Uses multi-threading for efficient vLLM API calls
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 3,
        compute_score=None,
        eval: bool = False,
        reward_fn_key: str = "data_source",
        num_threads: int = 8,
        enable_info_reward: bool = True,
        vllm_api_base: str = "http://10.24.1.118:9000/v1",
        vllm_model: str = "qwen2.5-7b-instruct" ,
        info_timeout: float = 30.0,
        score_source: str = "em",
        answer_candidates: Optional[List[str]] = None,
        info_reward_weight: float = 1.0,
        ppl_threshold: float = 100.0,
        delta_phi_scale: float = 0.5, # 1.0
        info_reward_start_step: int = 0,
        info_reward_warmup_steps: int = 0,
        verbose: bool = False
    ) -> None:
        """
        Initialize the InfoRewardLLMManager.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging.
            compute_score: A function to compute the main task reward score.
            eval: Whether this is evaluation mode.
            reward_fn_key: The key used to access the data source in the non-tensor batch data.
            num_threads: Number of threads for parallel vLLM API calls.
            enable_info_reward: Whether to enable information reward computation.
            vllm_api_base: vLLM API base URL.
            vllm_model: vLLM model name.
            info_timeout: Timeout for vLLM API calls.
            score_source: Score source for main task computation.
            answer_candidates: Default answer candidates for information reward when sample-specific candidates are not available.
            info_reward_weight: Weight for information rewards in the final reward tensor.
            ppl_threshold: Initial perplexity threshold above which all info rewards are set to 0 (default: 100.0).
            delta_phi_scale: Linear scaling factor for ΔΦ to achieve ~[-0.8, 0.8] range, preserves PBRS (default: 0.03).
            info_reward_start_step: Training step to start enabling info rewards (default: 0, immediate).
            info_reward_warmup_steps: Number of steps to gradually ramp up info reward weight (default: 0, no warmup).
            verbose: Whether to enable verbose logging.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_threads = max(1, int(num_threads))
        self.enable_info_reward = bool(enable_info_reward)
        self.vllm_api_base = "http://10.24.1.118:9000/v1"
        self.vllm_model = "qwen2.5-7b-instruct"
        self.info_timeout = float(info_timeout)
        self.score_source = score_source
        self.info_reward_weight = float(info_reward_weight)
        self.ppl_threshold = float(ppl_threshold)
        self.delta_phi_scale = float(delta_phi_scale)
        self.info_reward_start_step = int(info_reward_start_step)
        self.info_reward_warmup_steps = int(info_reward_warmup_steps)
        self.verbose = verbose
        
        # Internal step counter for info reward scheduling
        self.current_step = 0
        
        # Default answer candidates if not provided
        self.answer_candidates = answer_candidates or [
            "The answer is correct and well-supported.",
            "The answer is partially correct but incomplete.",
            "The answer is incorrect or misleading.",
        ]
        
        # Validate vLLM setup
        if self.enable_info_reward and OpenAI is None:
            print("Warning: OpenAI package not found. Information reward computation will be disabled.")
            self.enable_info_reward = False
        
        if self.enable_info_reward and not self.vllm_api_base:
            print("Warning: vLLM API base not configured. Information reward computation will be disabled.")
            self.enable_info_reward = False
            
        if self.enable_info_reward:
            print(f"InfoRewardLLMManager initialized with vLLM at {self.vllm_api_base}")
            print(f"Using {len(self.answer_candidates)} default answer candidates, {self.num_threads} threads")
            print(f"Initial perplexity threshold set to {self.ppl_threshold:.1f} (all info rewards set to 0 if initial ppl > threshold)")
            print(f"ΔΦ scaling: {self.delta_phi_scale:.3f} (target range: ~[-0.8, 0.8]), weight: {self.info_reward_weight:.2f}")
            if self.info_reward_start_step > 0:
                print(f"Info reward schedule: start at step {self.info_reward_start_step}, warmup over {self.info_reward_warmup_steps} steps")
            else:
                print(f"Info reward enabled immediately (no step-based scheduling)")
    
    def _compute_info_reward_step_weight(self, current_step: int) -> float:
        """
        Compute the current step weight for info rewards based on training schedule.
        
        Args:
            current_step: Current training step
            
        Returns:
            Weight multiplier for info rewards (0.0 to 1.0)
        """
        if current_step < self.info_reward_start_step:
            # Before start step: completely disabled
            return 0.0
        elif self.info_reward_warmup_steps <= 0:
            # No warmup: immediately full weight
            return 1.0
        else:
            # Warmup phase: linear ramp up
            steps_since_start = current_step - self.info_reward_start_step
            if steps_since_start >= self.info_reward_warmup_steps:
                # After warmup: full weight
                return 1.0
            else:
                # During warmup: linear interpolation
                return float(steps_since_start) / float(self.info_reward_warmup_steps)
    
    def reset_step_counter(self) -> None:
        """Reset the internal step counter to 0."""
        self.current_step = 0
        if self.verbose:
            print(f"InfoRewardLLMManager: Step counter reset to 0")
    
    def get_current_step(self) -> int:
        """Get the current step count."""
        return self.current_step
    
    def _extract_answer_candidates(self, ground_truth: Any) -> List[str]:
        """
        Extract answer candidates from ground_truth['target'].
        
        Args:
            ground_truth: Ground truth data, may contain 'target' field
            
        Returns:
            List of answer candidate strings, or empty list if not found
        """
        if not isinstance(ground_truth, dict):
            return []
        
        target = ground_truth.get('target', None)
        if target is None:
            return []
        
        # Handle numpy array
        try:
            import numpy as np
            if isinstance(target, np.ndarray):
                # Convert to list and then to strings
                target_list = target.tolist()
                if isinstance(target_list, list):
                    return [str(item) for item in target_list if item is not None]
                else:
                    return [str(target_list)]
        except ImportError:
            pass
        
        # Handle list
        if isinstance(target, list):
            return [str(item) for item in target if item is not None]
        
        # Handle single string
        if isinstance(target, str):
            return [target]
        
        # Handle other types (convert to string)
        try:
            return [str(target)]
        except Exception:
            return []
    
    def _print_segment_info_summary(self, info_results_by_index: dict) -> None:
        """Print a summary of segment-level information rewards."""
        print("\n=== SEGMENT INFO REWARD SUMMARY ===")
        
        total_segments = 0
        total_positive_rewards = 0
        total_negative_rewards = 0
        segment_rewards_list = []
        
        for sample_idx, (seg_ids, info_rewards, info_stats) in info_results_by_index.items():
            boundaries = _compute_segment_boundaries(seg_ids)
            num_boundaries = len(boundaries)
            total_segments += num_boundaries
            
            print(f"Sample {sample_idx}: {num_boundaries} segments, ΔΦ_total={info_stats.get('total_delta_phi', 0.0):.4f}")
            
            # Show per-segment rewards
            segment_rewards = {}
            for seg_idx, end_idx in enumerate(boundaries):
                if end_idx < len(info_rewards):
                    reward_val = info_rewards[end_idx]
                    segment_rewards[seg_idx] = reward_val
                    segment_rewards_list.append(reward_val)
                    
                    if reward_val > 0:
                        total_positive_rewards += 1
                    elif reward_val < 0:
                        total_negative_rewards += 1
            
            if segment_rewards:
                reward_str = ", ".join([f"seg{k}={v:.4f}" for k, v in segment_rewards.items()])
                print(f"  Per-segment: {reward_str}")
        
        # Overall statistics
        if segment_rewards_list:
            import numpy as np
            avg_reward = np.mean(segment_rewards_list)
            std_reward = np.std(segment_rewards_list)
            max_reward = max(segment_rewards_list)
            min_reward = min(segment_rewards_list)
            
            print(f"\nOverall stats: {total_segments} segments")
            print(f"  Positive rewards: {total_positive_rewards} ({100*total_positive_rewards/total_segments:.1f}%)")
            print(f"  Negative rewards: {total_negative_rewards} ({100*total_negative_rewards/total_segments:.1f}%)")
            print(f"  Zero rewards: {total_segments-total_positive_rewards-total_negative_rewards}")
            print(f"  Avg: {avg_reward:.4f}, Std: {std_reward:.4f}")
            print(f"  Range: [{min_reward:.4f}, {max_reward:.4f}]")
            
            # Aggregate delta_phi scale statistics across all samples
            all_delta_means = []
            all_delta_stds = []
            all_delta_abs_means = []
            all_delta_q05 = []
            all_delta_q25 = []
            all_delta_medians = []
            all_delta_q75 = []
            all_delta_q95 = []
            total_positive_deltas = 0
            total_negative_deltas = 0
            total_zero_deltas = 0
            
            for sample_idx, (seg_ids, info_rewards, info_stats) in info_results_by_index.items():
                if info_stats.get("num_boundaries", 0) > 0:
                    all_delta_means.append(info_stats.get("delta_phi_mean", 0.0))
                    all_delta_stds.append(info_stats.get("delta_phi_std", 0.0))
                    all_delta_abs_means.append(info_stats.get("delta_phi_abs_mean", 0.0))
                    all_delta_q05.append(info_stats.get("delta_phi_q05", 0.0))
                    all_delta_q25.append(info_stats.get("delta_phi_q25", 0.0))
                    all_delta_medians.append(info_stats.get("delta_phi_median", 0.0))
                    all_delta_q75.append(info_stats.get("delta_phi_q75", 0.0))
                    all_delta_q95.append(info_stats.get("delta_phi_q95", 0.0))
                    total_positive_deltas += info_stats.get("delta_phi_positive_count", 0)
                    total_negative_deltas += info_stats.get("delta_phi_negative_count", 0)
                    total_zero_deltas += info_stats.get("delta_phi_zero_count", 0)
            
            if all_delta_means:
                print(f"\nDelta Phi Scale Statistics:")
                print(f"  Avg delta_phi_mean: {np.mean(all_delta_means):.6f}")
                print(f"  Avg delta_phi_std: {np.mean(all_delta_stds):.6f}")
                print(f"  Avg delta_phi_abs_mean: {np.mean(all_delta_abs_means):.6f}")
                print(f"  Distribution (95% coverage):")
                print(f"    Q05: {np.mean(all_delta_q05):.6f}")
                print(f"    Q25: {np.mean(all_delta_q25):.6f}")
                print(f"    Q50 (median): {np.mean(all_delta_medians):.6f}")
                print(f"    Q75: {np.mean(all_delta_q75):.6f}")
                print(f"    Q95: {np.mean(all_delta_q95):.6f}")
                
                total_deltas = total_positive_deltas + total_negative_deltas + total_zero_deltas
                if total_deltas > 0:
                    print(f"  Total delta distribution: +{total_positive_deltas} -{total_negative_deltas} ±0{total_zero_deltas}")
                    print(f"  Percentage: +{100*total_positive_deltas/total_deltas:.1f}% -{100*total_negative_deltas/total_deltas:.1f}% ±0{100*total_zero_deltas/total_deltas:.1f}%")
        
        print("=" * 40)

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Process data and compute rewards.
        
        Args:
            data: DataProto containing batch data
            return_dict: Whether to return dict format
        """
        
        # Increment internal step counter
        self.current_step += 1
        
        # Compute step-based info reward weight
        step_weight = self._compute_info_reward_step_weight(self.current_step)
        effective_info_reward = self.enable_info_reward and (step_weight > 0.0)
        
        if self.verbose and self.enable_info_reward:
            print(f"Step {self.current_step}: info reward weight = {step_weight:.3f}, effective = {effective_info_reward}")
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        # Create reward_tensor with shape (bs, T) where final reward goes to valid_response_length - 1
        responses_shape = data.batch['responses'].shape  # (bs, T)
        bs, T = responses_shape
        reward_tensor = torch.zeros((bs, T), dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # Create stats tensor: [format_reward, f1, em, info_reward_total, num_boundaries, max_phi, min_phi, total_delta_phi, 
        #                      delta_phi_mean, delta_phi_std, delta_phi_max, delta_phi_min, delta_phi_abs_mean, 
        #                      delta_phi_positive_count, delta_phi_negative_count, delta_phi_zero_count,
        #                      delta_phi_q05, delta_phi_q25, delta_phi_median, delta_phi_q75, delta_phi_q95,
        #                      filtered_out, total_samples]
        stats_tensor = torch.zeros((bs, 24), dtype=torch.float32)

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
            # Even in fallback, try to minimize decode calls where possible
            prompt_strs_list = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch_valid_prompt_ids]
            response_strs_list = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in batch_valid_response_ids]

        # Batch convert tokens for segment detection (optimized for performance)
        if len(batch_valid_response_ids) > 1:
            print(f"Batch converting {len(batch_valid_response_ids)} token sequences...")
        
        batch_token_strings = []
        try:
            # Use batch decode for token strings - much simpler and more efficient!
            if hasattr(self.tokenizer, 'batch_decode'):
                # Prepare all single-token lists for batch decode
                all_single_tokens = []
                token_counts = []
                for response_ids in batch_valid_response_ids:
                    single_token_lists = [[int(t)] for t in response_ids]
                    all_single_tokens.extend(single_token_lists)
                    token_counts.append(len(response_ids))
                
                if self.verbose and len(all_single_tokens) > 100:
                    print(f"Batch decoding {len(all_single_tokens)} individual tokens for boundary detection")
                
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
        info_reward_tasks = []
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
            
            # Information reward task (if enabled)
            if self.enable_info_reward:
                # Extract answer candidates from ground_truth['target'] if available
                sample_answer_candidates = self._extract_answer_candidates(metadata['ground_truth'])
                if not sample_answer_candidates:
                    # Fallback to default candidates
                    raise ValueError("No answer candidates found")
                else:
                    pass
                
                info_reward_tasks.append((
                    i, prompt_str, response_str, sample_answer_candidates,
                    token_strings, self.vllm_model, self.info_timeout, self.verbose
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

        # Process information reward tasks (multi-threaded)
        info_results_by_index = {}
        if effective_info_reward and info_reward_tasks:
            print(f"Computing information rewards for {len(info_reward_tasks)} samples using {self.num_threads} threads...")
            
            # Split tasks into chunks for threading
            chunk_size = max(1, len(info_reward_tasks) // self.num_threads)
            chunks = [info_reward_tasks[i:i + chunk_size] for i in range(0, len(info_reward_tasks), chunk_size)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [
                    executor.submit(_process_info_reward_batch, chunk, self.vllm_api_base, self.info_timeout, 
                                  self.ppl_threshold, self.delta_phi_scale)
                    for chunk in chunks
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_results = future.result()
                        info_results_by_index.update(chunk_results)
                    except Exception as e:
                        print(f"Error in information reward computation: {e}")
            
            # Print segment-level info reward summary
            if info_results_by_index and self.verbose:
                self._print_segment_info_summary(info_results_by_index)

        # Aggregate rewards into tensor
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())

            # Add information rewards per token to sequence positions
            adjusted_info_rewards = []  # Initialize for stats calculation
            if i in info_results_by_index:
                seg_ids, info_rewards, info_stats, delta_phi = info_results_by_index[i]
                if info_rewards and valid_response_length > 1:  # Need at least 2 tokens (one for info reward, one for final)
                    # Get response_mask for this sample
                    if "response_mask" in data_item.batch:
                        response_mask_tensor = data_item.batch["response_mask"][:valid_response_length]
                        response_mask = response_mask_tensor.cpu().tolist()
                    else:
                        # Fallback: compute from attention_mask if response_mask not available
                        full_attention_mask = data_item.batch["attention_mask"]
                        response_attention_mask = full_attention_mask[prompt_length:prompt_length + valid_response_length]
                        response_mask = response_attention_mask.cpu().tolist()
                    
                    # Directly place rewards at last valid token in each segment
                    adjusted_info_rewards = _per_token_info_rewards_with_mask(
                        seg_ids, delta_phi, response_mask
                    )
                    
                    max_pos = min(len(adjusted_info_rewards), valid_response_length - 1)  # Avoid overwriting final reward
                    for pos in range(max_pos):
                        # Apply both configured weight and step-based weight
                        effective_weight = self.info_reward_weight * step_weight
                        reward_tensor[i, pos] += float(adjusted_info_rewards[pos]) * effective_weight
                
                # Update stats tensor with info reward stats
                stats_tensor[i, 4] = float(info_stats.get("num_boundaries", 0))
                stats_tensor[i, 5] = float(info_stats.get("max_phi", 0.0))
                stats_tensor[i, 6] = float(info_stats.get("min_phi", 0.0))
                stats_tensor[i, 7] = float(info_stats.get("total_delta_phi", 0.0))
                
                # Add delta_phi scale statistics
                stats_tensor[i, 8] = float(info_stats.get("delta_phi_mean", 0.0))
                stats_tensor[i, 9] = float(info_stats.get("delta_phi_std", 0.0))
                stats_tensor[i, 10] = float(info_stats.get("delta_phi_max", 0.0))
                stats_tensor[i, 11] = float(info_stats.get("delta_phi_min", 0.0))
                stats_tensor[i, 12] = float(info_stats.get("delta_phi_abs_mean", 0.0))
                stats_tensor[i, 13] = float(info_stats.get("delta_phi_positive_count", 0))
                stats_tensor[i, 14] = float(info_stats.get("delta_phi_negative_count", 0))
                stats_tensor[i, 15] = float(info_stats.get("delta_phi_zero_count", 0))
                stats_tensor[i, 16] = float(info_stats.get("delta_phi_q05", 0.0))
                stats_tensor[i, 17] = float(info_stats.get("delta_phi_q25", 0.0))
                stats_tensor[i, 18] = float(info_stats.get("delta_phi_median", 0.0))
                stats_tensor[i, 19] = float(info_stats.get("delta_phi_q75", 0.0))
                stats_tensor[i, 20] = float(info_stats.get("delta_phi_q95", 0.0))
                stats_tensor[i, 21] = float(1.0 if info_stats.get("filtered_out", False) else 0.0)  # filtered_out flag
                stats_tensor[i, 22] = 1.0  # total_samples counter

            # Main score goes to the final valid position (like naive.py)
            main_score = main_score_by_index.get(i, 0.0)
            main_extra = main_extra_by_index.get(i, {})
            main_stats = main_stats_by_index.get(i, {})
            
            # Final reward goes to the last valid token position (like naive.py)
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = float(main_score)

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
            stats_tensor[i, 3] = float(sum(adjusted_info_rewards) if adjusted_info_rewards else 0.0)
            
            # Set total_samples counter for all samples (including those without info rewards)
            stats_tensor[i, 22] = 1.0

            data_source = batch_metadata[i]['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # Debug printing (commented out like naive_llm.py)
            # if already_print_data_sources[data_source] < self.num_examine:
            #     already_print_data_sources[data_source] += 1
            #     print(f"\n[Sample {i}] Data source: {data_source}")
            #     print(f"[Sample {i}] Main score: {main_score:.4f}")
            #     if i in info_results_by_index:
            #         info_stats = info_results_by_index[i][2]
            #         print(f"[Sample {i}] Info reward stats: {info_stats}")
            #         # Print per-segment info reward details
            #         seg_ids, info_rewards, _, _ = info_results_by_index[i]
            #         boundaries = _compute_segment_boundaries(seg_ids)
            #         print(f"[Sample {i}] Segment boundaries: {boundaries}")
            #         segment_rewards = {}
            #         for seg_idx, end_idx in enumerate(boundaries):
            #             if end_idx < len(info_rewards):
            #                 segment_rewards[seg_idx] = info_rewards[end_idx]
            #         print(f"[Sample {i}] Per-segment info rewards: {segment_rewards}")
            #     if isinstance(main_extra, dict) and main_extra:
            #         for key, value in main_extra.items():
            #             if key != "score":  # Avoid duplicate
            #                 print(f"[Sample {i}] {key}: {value}")

        # Save sample data for debugging and analysis with 1/30 probability
        if True:
            try:
                # Create mapping from sample index to strings
                prompt_strs_dict = {}
                response_strs_dict = {}
                for idx, metadata in enumerate(batch_metadata):
                    i = metadata['index']
                    prompt_strs_dict[i] = prompt_strs_list[idx]
                    response_strs_dict[i] = response_strs_list[idx]
                
                _save_sample_data_info_reward(
                    data, reward_tensor, main_extra_by_index, info_results_by_index, 
                    prompt_strs_dict, response_strs_dict, self.tokenizer, max_samples=1
                )
            except Exception as e:
                print(f"Warning: Failed to save sample data: {e}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "info_stats_tensor": stats_tensor,
                "info_results_by_index": info_results_by_index,
                "info_flag": True,
            }
        else:
            # Return both reward and stats tensors (like naive_llm.py)
            return reward_tensor, stats_tensor
