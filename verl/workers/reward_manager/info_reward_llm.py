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
import json
    
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
# Boundary detection: only use </tool_response> to define complete segments
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
    
    Note: We only use </tool_response> to define segment boundaries for info reward calculation.
    The actual reward placement will target </tool_call> within each segment.
    
    This is adapted from naive_llm.py to work with information reward boundaries.
    """
    resp_len = len(token_strings)
    if resp_len == 0:
        return []

    # Only use </tool_response> to define complete segments
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
    ctx_text += "assistant\n"
    open_tag  =  "<answer>"
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
    length_norm: bool = True,
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
            model=model,  # Use the model parameter instead of hardcoded value
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
        if verbose or True:  # Always log API failures for debugging
            print(f"[DEBUG] vLLM API call failed for model={model}: {e}")
            print(f"[DEBUG] Prompt length: {len(prompt)}, Answer: {answer_text[:50]}...")
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
            if verbose or True:  # Always log when no tokens selected
                print(f"[DEBUG] No tokens selected within answer span for: {answer_text[:50]}...")
                print(f"[DEBUG] Answer span: [{ans_start}, {ans_end}), prompt length: {len(prompt)}")
                print(f"[DEBUG] Text offsets available: {text_offsets is not None}, echoed_text length: {len(echoed_text)}")
                if text_offsets:
                    print(f"[DEBUG] First few text_offsets: {text_offsets[:5]}")
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
    length_norm: bool = True,
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
        if verbose:
            print(f"[DEBUG] No </tool_response> boundaries found in response (length: {len(response_text)} chars)")
            print(f"[DEBUG] Response preview: {response_text[:200]}...")
            # Check which tags are present
            has_tool_response = '</tool_response>' in response_text.lower()
            has_tool_call = '</tool_call>' in response_text.lower()
            print(f"[DEBUG] Contains </tool_response>: {has_tool_response}, Contains </tool_call>: {has_tool_call}")
            print(f"[DEBUG] Note: Segments are defined by </tool_response>, rewards will be placed at </tool_call>")
        return InfoRewardResult(
            boundaries=[BoundaryScore(i=0, phi=0.0, per_cand_logp=[0.0] * len(answer_candidates))],
            delta_phi=[]
        )
    
    # Build contexts: initial + after each boundary
    if verbose:
        print(f"[DEBUG] Found {len(boundaries_inclusive)} </tool_response> segment boundaries at positions: {boundaries_inclusive}")
        print(f"[DEBUG] Info rewards will be calculated at these boundaries, but placed at </tool_call> tokens within each segment")
    
    contexts = [prompt_text]
    for end_pos in boundaries_inclusive:
        ctx_i = prompt_text + response_text[: end_pos + 1]
        contexts.append(ctx_i)

    # Check initial perplexity threshold on prompt-only context
    # If initial ppl is too high, return zero rewards for entire sample
    initial_ctx = contexts[0]  # prompt_text
    initial_scores_for_threshold = []
    for answer in answer_candidates:
        score = _score_answer_with_vllm(
            initial_ctx, answer, vllm_client, model,
            length_norm=True,  # Use length-normalized for perplexity threshold check
            timeout=timeout, verbose=verbose
        )
        initial_scores_for_threshold.append(score)
    
    # Check if initial score indicates poor quality (corrected logic)
    max_initial_score = max(initial_scores_for_threshold) if initial_scores_for_threshold else float('-inf')
    # Convert threshold to log-prob: if ppl_threshold=100, then log_prob_threshold = log(1/100) = -4.6
    log_prob_threshold = np.log(1.0 / ppl_threshold) if ppl_threshold > 0 else float('-inf')
    
    # Always log initial scores for debugging
    if verbose or max_initial_score == 0.0:
        print(f"[DEBUG] Initial scores for threshold check: {initial_scores_for_threshold}")
        print(f"[DEBUG] Max initial score: {max_initial_score:.4f}, threshold: {log_prob_threshold:.4f} (ppl_threshold={ppl_threshold})")
        if max_initial_score == 0.0:
            print(f"[DEBUG] WARNING: All initial scores are 0.0 - vLLM API may have failed or no tokens matched")
    
    # if max_initial_score < log_prob_threshold:
    #     if verbose:
    #         print(f"[DEBUG] FILTERED by ppl_threshold: Initial max log-prob {max_initial_score:.4f} < threshold {log_prob_threshold:.4f} (ppl > {ppl_threshold})")
    #         print(f"[DEBUG] Setting all info rewards to 0 for this sample")
    #     # Return zero rewards for entire sample
    #     zero_boundaries = []
    #     for i in range(len(contexts)):
    #         zero_boundaries.append(BoundaryScore(i=i, phi=0.0, per_cand_logp=[0.0] * len(answer_candidates)))
    #     return InfoRewardResult(boundaries=zero_boundaries, delta_phi=[0.0] * (len(contexts) - 1))

    boundary_scores: List[BoundaryScore] = []

    # Score each context against all answer candidates
    # IMPORTANT: Use consistent length_norm=False for all Φ calculations to ensure ΔΦ comparability
    for i, ctx in enumerate(contexts):
        per_cand = []
        for answer in answer_candidates:
            # Use joint log probability (sum logp) for all boundaries including initial
            score = _score_answer_with_vllm(
                ctx, answer, vllm_client, model,
                length_norm=True,  # Consistent: no length normalization for Φ calculation
                timeout=timeout, verbose=verbose
            )
            per_cand.append(score)

        # Debug: log if all scores are 0
        if verbose or (per_cand and all(s == 0.0 for s in per_cand)):
            print(f"[DEBUG] Boundary {i}: per_cand scores = {per_cand}")

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
                if verbose or m == 0.0:
                    print(f"[DEBUG] Boundary {i}: max score is not finite (m={m}), setting phi_i=0.0")
                phi_i = 0.0
        else:
            # Fallback to 0.0 when no candidates (should not happen in normal flow: ensure candidates or use default fallback)
            if verbose:
                print(f"[DEBUG] Boundary {i}: no candidates, setting phi_i=0.0")
            phi_i = 0.0

        if verbose or phi_i == 0.0:
            print(f"[DEBUG] Boundary {i}: phi_i = {phi_i:.6f}")

        boundary_scores.append(BoundaryScore(i=i, phi=phi_i, per_cand_logp=per_cand))

    phis = np.asarray([b.phi for b in boundary_scores], dtype=np.float64)

    # Return original (unscaled) delta_phi - scaling will be done in main thread
    delta_arr = phis[1:] - phis[:-1] if phis.size >= 2 else np.asarray([], dtype=np.float64)
    
    if delta_arr.size:
        delta_raw = delta_arr.tolist()  # Return raw delta_phi (no scaling here)
    else:
        delta_raw = []

    return InfoRewardResult(boundaries=boundary_scores, delta_phi=delta_raw)
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


def _per_token_info_rewards_with_mask(
    seg_ids: List[int], 
    delta_phi: List[float], 
    response_mask: List[int],
    token_strings: Optional[List[str]] = None
) -> List[float]:
    """
    Create a per-token reward vector by assigning delta_phi values at the <|im_end|> token
    following </tool_call> in each segment.
    
    Strategy: 
    - Segments are defined by </tool_response> boundaries (for info reward calculation)
    - But rewards are placed at </tool_call> + <|im_end|> within each segment
    - This allows placing rewards at the point where agent completes its tool call,
      while calculating info gain based on the complete tool response.
    
    Args:
        seg_ids: Segment IDs for each token (computed based on </tool_response> boundaries)
        delta_phi: Information reward deltas for each segment
        response_mask: Response mask indicating valid tokens (1=valid, 0=invalid)
        token_strings: Token strings for identifying specific tokens (optional)
    
    Returns:
        Per-token reward vector with rewards placed only at <|im_end|> after </tool_call>
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

    # For each segment, place reward ONLY at <|im_end|> after </tool_response> or </tool_call>
    for seg_idx, segment_end_idx in enumerate(boundaries):
        seg_reward = per_segment_info[seg_idx]
        if seg_reward == 0.0:
            continue
            
        # Find the segment start
        if seg_idx == 0:
            segment_start_idx = 0
        else:
            segment_start_idx = boundaries[seg_idx - 1] + 1
        
        # Find </tool_call> followed by <|im_end|> in this segment
        # Note: Segment boundaries are defined by </tool_response>, but rewards are placed at </tool_call>
        reward_pos = None
        pattern_found = False
        if token_strings and len(token_strings) >= resp_len:
            for pos in range(segment_start_idx, min(segment_end_idx + 1, resp_len - 1)):
                current_token = token_strings[pos]
                next_token = token_strings[pos + 1] if pos + 1 < len(token_strings) else None
                
                # Only look for </tool_call> (not </tool_response>) for reward placement
                is_tool_call = (current_token == '</tool_call>')
                
                if is_tool_call and next_token == '<|im_end|>':
                    pattern_found = True
                    # Check if <|im_end|> is valid (response_mask=1)
                    if not response_mask or (pos + 1 < len(response_mask) and response_mask[pos + 1] == 1):
                        reward_pos = pos + 1  # Place reward at <|im_end|>
                        # print(f"[DEBUG] Segment {seg_idx}: Found {current_token} + <|im_end|> at pos {pos}, placing reward at {reward_pos}")
                        break
                    else:
                        print(f"[DEBUG] Segment {seg_idx}: Found {current_token} + <|im_end|> at pos {pos} but <|im_end|> is masked")
        
        # Debug: report if pattern was not found
        if not pattern_found and seg_reward != 0.0:
            print(f"[DEBUG] Segment {seg_idx}: Pattern </tool_call> + <|im_end|> NOT found in range [{segment_start_idx}, {segment_end_idx}]")
            # Sample some tokens for debugging
            if token_strings and len(token_strings) >= resp_len:
                sample_range = range(max(0, segment_start_idx - 2), min(segment_end_idx + 3, len(token_strings)))
                sample_tokens = [f"{i}:{repr(token_strings[i])}" for i in sample_range]
                print(f"[DEBUG] Segment {seg_idx}: Sample tokens around segment: {sample_tokens}")
        
        # Only place reward if the pattern was found
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
    ppl_threshold: float = 100.0
) -> Dict[int, Tuple[List[int], List[float], dict]]:
    """
    Process a batch of information reward tasks in a single thread.
    
    Args:
        batch_tasks: List of (index, prompt_text, response_text, answer_candidates, token_strings, model, timeout, verbose)
        vllm_api_base: vLLM API base URL
        timeout: Request timeout
        ppl_threshold: Initial perplexity threshold above which all info rewards are set to 0
        
    Returns:
        Dict mapping index to (seg_ids, info_rewards_unscaled, stats, delta_phi_raw)
        Note: info_rewards_unscaled and delta_phi_raw are NOT scaled - scaling happens in main thread
    """
    results = {}
    
    if not batch_tasks or not vllm_api_base:
        return results
    
    # Ensure base_url ends with /v1 for OpenAI client compatibility
    base_url_normalized = vllm_api_base.rstrip('/')
    if not base_url_normalized.endswith('/v1'):
        base_url_normalized = base_url_normalized + '/v1'
    
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
            base_url=base_url_normalized,
            http_client=http_client,
        )
    except Exception as e:
        print(f"Failed to initialize vLLM client: {e}")
        return results
    
    for task in batch_tasks:
        (index, prompt_text, response_text, answer_candidates, 
         token_strings, model, timeout, verbose) = task
        
        # Debug: log answer_candidates
        if verbose or not answer_candidates:
            print(f"[DEBUG] Sample {index}: answer_candidates count = {len(answer_candidates) if answer_candidates else 0}")
            if answer_candidates:
                print(f"[DEBUG] Sample {index}: answer_candidates = {[c[:50] + '...' if len(c) > 50 else c for c in answer_candidates]}")
            else:
                print(f"[DEBUG] Sample {index}: WARNING - No answer_candidates provided!")
        
        try:
            # Compute information reward (returns raw/unscaled delta_phi)
            info_result = _compute_info_reward_for_sample(
                prompt_text=prompt_text,
                response_text=response_text,
                answer_candidates=answer_candidates,
                vllm_client=vllm_client,
                model=model,
                length_norm=True,
                timeout=timeout,
                verbose=verbose,
                ppl_threshold=ppl_threshold
            )
            
            # Build segment IDs from tokens
            seg_ids = _build_seg_ids_from_tokens(token_strings)
            
            # Convert delta_phi to per-token rewards (unscaled - scaling happens in main thread)
            # These are placeholder positions, actual values will be scaled in main thread
            info_rewards = _per_token_info_rewards_simple(seg_ids, info_result.delta_phi)
            
            # Collect stats
            # Extract all phi values from boundaries
            phi_values = [float(b.phi) for b in info_result.boundaries] if info_result.boundaries else []
            
            # num_boundaries includes initial boundary (i=0), so actual tool use count = len(delta_phi)
            # tool_use_count represents the actual number of tool uses (excluding initial boundary)
            tool_use_count = len(info_result.delta_phi) if info_result.delta_phi else 0
            
            # Calculate mean phi
            mean_phi = float(np.mean(phi_values)) if phi_values else 0.0
            
            stats = {
                "num_boundaries": len(info_result.boundaries),  # Includes initial boundary (i=0)
                "tool_use_count": tool_use_count,  # Actual number of tool uses (excludes initial boundary)
                "total_delta_phi": sum(info_result.delta_phi) if info_result.delta_phi else 0.0,
                "max_phi": max(phi_values) if phi_values else 0.0,
                "min_phi": min(phi_values) if phi_values else 0.0,
                "mean_phi": mean_phi,  # Mean of all phi values
                "final_phi": float(info_result.boundaries[-1].phi) if info_result.boundaries else 0.0,
                "phi_values": phi_values,  # Save all phi values for each boundary
            }
            
            # Add delta_phi scale statistics (RAW/UNSCALED values)
            # NOTE: These will be RECOMPUTED in main thread after scaling is applied
            # Main thread will replace these with scaled values for consistency
            if info_result.delta_phi:
                delta_arr = np.array(info_result.delta_phi)
                # Calculate quantiles with wider coverage (5th, 25th, 50th, 75th, 95th percentiles)
                quantiles = np.percentile(delta_arr, [5, 25, 50, 75, 95])
                stats.update({
                    "delta_phi_mean": float(np.mean(delta_arr)),
                    "delta_phi_std": float(np.std(delta_arr)),
                    "delta_phi_max": float(np.max(delta_arr)),
                    "delta_phi_min": float(np.min(delta_arr)),
                    "delta_phi_abs_mean": float(np.mean(np.abs(delta_arr))),  # Will be replaced with scaled value
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
            stats = {
                "num_boundaries": 0,
                "tool_use_count": 0,  # Actual number of tool uses
                "total_delta_phi": 0.0, 
                "max_phi": 0.0, 
                "min_phi": 0.0,
                "mean_phi": 0.0,  # Mean of all phi values
                "final_phi": 0.0,
                "phi_values": [],  # Empty phi values on error
            }
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
            tool_use_count = info_stats.get("tool_use_count", 0)
            if len(unique_segments) >= 1 and tool_use_count > 0:
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
                    f.write(f"Tool use count: {info_stats.get('tool_use_count', 0)} (actual tool uses)\n")
                    f.write(f"Info reward boundaries: {info_stats.get('num_boundaries', 0)} (includes initial boundary)\n")
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
                        if key == "phi_values":
                            # Skip phi_values here, will print separately
                            continue
                        elif key.startswith("delta_phi"):
                            if "count" in key:
                                f.write(f"{key}: {value}\n")
                            else:
                                f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    
                    # Print all phi values for each boundary
                    if "phi_values" in info_stats:
                        phi_values = info_stats["phi_values"]
                        f.write(f"\n=== PHI VALUES (ALL BOUNDARIES) ===\n")
                        f.write(f"Total boundaries: {len(phi_values)}\n")
                        for i, phi in enumerate(phi_values):
                            f.write(f"  Boundary {i}: Φ = {phi:.6f}\n")
                        # Also print delta_phi for comparison
                        if delta_phi and len(phi_values) > 1:
                            f.write(f"\nDelta Phi values (ΔΦ = Φ_i - Φ_{i-1}):\n")
                            for j, dphi in enumerate(delta_phi):
                                if j + 1 < len(phi_values):
                                    f.write(f"  ΔΦ_{j+1} = Φ_{j+1} - Φ_{j} = {phi_values[j+1]:.6f} - {phi_values[j]:.6f} = {dphi:.6f}\n")
                    
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


def _save_phi_values_json(info_results_by_index: Dict[int, Tuple], step: int, output_dir: str = "phi_values") -> None:
    """
    Save all phi values for all responses to a JSON file named by step.
    
    Args:
        info_results_by_index: Dictionary mapping sample index to (seg_ids, info_rewards, info_stats, delta_phi)
        step: Current training step (used for filename)
        output_dir: Directory to save JSON files (default: "phi_values")
    """
    try:
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect phi values for all responses
        # Only save samples with actual computed phi values (not all zeros)
        # Filter out samples that:
        #   - Have no boundaries found (num_boundaries=1, all phi=0.0)
        #   - Were filtered by ppl_threshold (all phi=0.0)
        phi_values_data = {}
        
        for sample_idx, (_, _, info_stats, delta_phi) in info_results_by_index.items():
            phi_values = info_stats.get("phi_values", [])
            if not phi_values:
                continue
            
            num_boundaries = len(phi_values)
            max_phi = info_stats.get("max_phi", 0.0)
            
            # Check if there are actual non-zero phi values
            # This filters out:
            #   - Samples with no boundaries found (num_boundaries=1, all phi=0.0)
            #   - Samples filtered by ppl_threshold (all phi=0.0)
            has_nonzero_phi = abs(max_phi) > 1e-6 or any(abs(phi) > 1e-6 for phi in phi_values)
            
            # Only save samples with actual computed non-zero phi values
            if not has_nonzero_phi:
                continue
            
            # Save samples with actual computed phi values
            phi_values_data[sample_idx] = {
                "phi_values": phi_values,
                "delta_phi": delta_phi,
                "num_boundaries": num_boundaries,
                "final_phi": float(phi_values[-1]) if phi_values else 0.0,
                "max_phi": float(max_phi),
                "min_phi": float(info_stats.get("min_phi", 0.0)),
            }
        
        # Save to JSON file
        if phi_values_data:
            filename = os.path.join(output_dir, f"phi_values_step_{step}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(phi_values_data, f, indent=2, ensure_ascii=False)
            print(f"Saved phi values for {len(phi_values_data)} samples to {filename}")
        else:
            print(f"No phi values to save for step {step}")
            
    except Exception as e:
        print(f"Failed to save phi values JSON: {e}")
        import traceback
        traceback.print_exc()


def _save_rollout_data_with_rewards(
    prompt_strs_list: List[str],
    response_strs_list: List[str],
    info_results_by_index: Dict[int, Tuple],
    main_score_by_index: Dict[int, float],
    batch_metadata: List[Dict],
    step: int,
    output_dir: str = "rollout_data",
    max_samples: Optional[int] = None,
    save_all: bool = False
) -> None:
    """
    Save rollout data (prompt, response) along with phi values and info rewards to JSON file.
    
    Args:
        prompt_strs_list: List of prompt strings for each sample
        response_strs_list: List of response strings for each sample
        info_results_by_index: Dictionary mapping sample index to (seg_ids, info_rewards, info_stats, delta_phi)
        main_score_by_index: Dictionary mapping sample index to main task score
        batch_metadata: List of metadata dictionaries for each sample
        step: Current training step (used for filename)
        output_dir: Directory to save JSON files (default: "rollout_data")
        max_samples: Maximum number of samples to save (None = save all, only applies if save_all=False)
        save_all: If True, save all samples. If False, only save samples with info rewards.
    """
    try:
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
        rollout_data = []
        saved_count = 0
        
        for idx, (prompt_str, response_str) in enumerate(zip(prompt_strs_list, response_strs_list)):
            # Get sample index from metadata
            if idx < len(batch_metadata):
                sample_idx = batch_metadata[idx].get('index', idx)
                metadata = batch_metadata[idx]
            else:
                sample_idx = idx
                metadata = {}
            
            # Decide whether to save this sample
            if not save_all:
                # Only save samples with info rewards
                if sample_idx not in info_results_by_index:
                    continue
                
                seg_ids, info_rewards, info_stats, delta_phi = info_results_by_index[sample_idx]
                tool_use_count = info_stats.get("tool_use_count", 0)
                
                # Only save samples with actual tool use
                if tool_use_count == 0:
                    continue
            
            # Limit number of samples if specified
            if max_samples is not None and saved_count >= max_samples:
                break
            
            # Collect info reward data if available
            info_reward_data = None
            if sample_idx in info_results_by_index:
                seg_ids, info_rewards, info_stats, delta_phi = info_results_by_index[sample_idx]
                phi_values = info_stats.get("phi_values", [])
                
                info_reward_data = {
                    "phi_values": [float(phi) for phi in phi_values],
                    "delta_phi": [float(d) for d in delta_phi],
                    "info_rewards": [float(r) for r in info_rewards],
                    "seg_ids": seg_ids,
                    "num_boundaries": len(phi_values),
                    "tool_use_count": info_stats.get("tool_use_count", 0),
                    "total_delta_phi": float(info_stats.get("total_delta_phi", 0.0)),
                    "max_phi": float(info_stats.get("max_phi", 0.0)),
                    "min_phi": float(info_stats.get("min_phi", 0.0)),
                    "mean_phi": float(info_stats.get("mean_phi", 0.0)),
                    "final_phi": float(info_stats.get("final_phi", 0.0)),
                    "delta_phi_mean": float(info_stats.get("delta_phi_mean", 0.0)),
                    "delta_phi_std": float(info_stats.get("delta_phi_std", 0.0)),
                    "delta_phi_max": float(info_stats.get("delta_phi_max", 0.0)),
                    "delta_phi_min": float(info_stats.get("delta_phi_min", 0.0)),
                    "delta_phi_abs_mean": float(info_stats.get("delta_phi_abs_mean", 0.0)),
                }
            
            # Build sample data
            sample_data = {
                "sample_index": sample_idx,
                "step": step,
                "prompt": prompt_str,
                "response": response_str,
                "main_score": float(main_score_by_index.get(sample_idx, 0.0)),
                "data_source": metadata.get("data_source", ""),
                "ground_truth": metadata.get("ground_truth", ""),
                "extra_info": metadata.get("extra_info", {}),
            }
            
            # Add info reward data if available
            if info_reward_data:
                sample_data["info_reward"] = info_reward_data
            
            rollout_data.append(sample_data)
            saved_count += 1
        
        # Save to JSON file
        if rollout_data:
            filename = os.path.join(output_dir, f"rollout_step_{step}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(rollout_data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(rollout_data)} rollout samples with rewards to {filename}")
        else:
            print(f"No rollout data to save for step {step}")
            
    except Exception as e:
        print(f"Failed to save rollout data JSON: {e}")
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


@register("info_reward_llm")
class InfoRewardLLMManager(AbstractRewardManager):
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
        vllm_api_base: str = "http://10.24.24.49:9000/v1",
        vllm_model: str = "qwen2.5-7b-instruct" ,
        info_timeout: float = 30.0,
        score_source: str = "em",
        answer_candidates: Optional[List[str]] = None,
        info_reward_weight: float = 1.0,
        ppl_threshold: float = 100.0,
        delta_phi_scale: float = 0.02, # 0.05
        info_reward_start_step: int = 0,
        info_reward_warmup_steps: int = 0,
            enable_adaptive_alpha: bool = False,
            alpha_update_rate: float = 0.1,
            alpha_min: float = 0.005,
            alpha_max: float = 0.5,
            enable_scale_control: bool = True,
            target_scale_min: float = 0.02,
            target_scale_max: float = 0.2,
            scale_update_rate: float = 0.1,
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
            enable_adaptive_alpha: Whether to enable adaptive alpha (delta_phi_scale) adjustment based on teacher quality.
            alpha_update_rate: Learning rate for updating delta_phi_scale based on alpha_ema (default: 0.1).
            alpha_min: Minimum value for delta_phi_scale when using adaptive alpha (default: 0.001).
            alpha_max: Maximum value for delta_phi_scale when using adaptive alpha (default: 0.1).
            enable_scale_control: Whether to enable scale control to keep info rewards in target range (default: True).
            target_scale_min: Minimum target scale for info rewards (default: 0.1).
            target_scale_max: Maximum target scale for info rewards (default: 0.4).
            scale_update_rate: Learning rate for updating delta_phi_scale based on observed scale (default: 0.1).
            verbose: Whether to enable verbose logging.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_threads = max(1, int(num_threads))
        self.enable_info_reward = bool(enable_info_reward)
        # Ensure base_url ends with /v1 for OpenAI client compatibility
        base_url_normalized = vllm_api_base.rstrip('/') if vllm_api_base else ""
        if base_url_normalized and not base_url_normalized.endswith('/v1'):
            base_url_normalized = base_url_normalized + '/v1'
        self.vllm_api_base = base_url_normalized
        self.vllm_model = vllm_model
        self.info_timeout = float(info_timeout)
        self.score_source = score_source
        self.info_reward_weight = float(info_reward_weight)
        self.ppl_threshold = float(ppl_threshold)
        self.delta_phi_scale = float(delta_phi_scale)
        self.base_delta_phi_scale = float(delta_phi_scale)  # Store base value for adaptive adjustment
        self.info_reward_start_step = int(info_reward_start_step)
        self.info_reward_warmup_steps = int(info_reward_warmup_steps)
        self.enable_adaptive_alpha = bool(enable_adaptive_alpha)
        self.alpha_update_rate = float(alpha_update_rate)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.enable_scale_control = bool(enable_scale_control)
        self.target_scale_min = float(target_scale_min)
        self.target_scale_max = float(target_scale_max)
        self.scale_update_rate = float(scale_update_rate)
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
        # Test vLLM API connection if enabled
        if self.enable_info_reward:
            try:
                import httpx
                http_client = httpx.Client(
                    limits=httpx.Limits(
                        max_keepalive_connections=10,
                        max_connections=50,
                        keepalive_expiry=30.0
                    ),
                    timeout=httpx.Timeout(self.info_timeout),
                )
                vllm_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
                    base_url=self.vllm_api_base,  # Already normalized to end with /v1
                    http_client=http_client,
                )
                
                # Send a simple test request
                test_prompt = "Hello, this is a test prompt."
                test_response = vllm_client.completions.create(
                    model=self.vllm_model,
                    prompt=test_prompt,
                    max_tokens=0,
                    echo=True,
                    logprobs=1
                )
                print(f"vLLM API test successful - model: {self.vllm_model}")
                
            except Exception as e:
                print("vllm base url: ", self.vllm_api_base)
                print(f"Warning: vLLM API test failed ({str(e)}). Info reward will still be enabled but may fail at runtime.")
                # Don't disable info reward - allow runtime retry
                # self.enable_info_reward = False
            
        if self.enable_info_reward:
            print(f"InfoRewardLLMManager initialized with vLLM at {self.vllm_api_base}")
            print(f"Using {len(self.answer_candidates)} default answer candidates, {self.num_threads} threads")
            print(f"Initial perplexity threshold set to {self.ppl_threshold:.1f} (all info rewards set to 0 if initial ppl > threshold)")
            print(f"ΔΦ scaling: {self.delta_phi_scale:.3f} (target range: ~[-0.8, 0.8]), weight: {self.info_reward_weight:.2f}")
            if self.enable_adaptive_alpha:
                print(f"Adaptive alpha enabled: update_rate={self.alpha_update_rate:.3f}, range=[{self.alpha_min:.6f}, {self.alpha_max:.6f}]")
            if self.enable_scale_control:
                print(f"Scale control enabled: target=[{self.target_scale_min:.2f}, {self.target_scale_max:.2f}], update_rate={self.scale_update_rate:.3f}")
            if self.info_reward_start_step > 0:
                print(f"Info reward schedule: start at step {self.info_reward_start_step}, warmup over {self.info_reward_warmup_steps} steps")
            else:
                print(f"Info reward enabled immediately (no step-based scheduling)")

        # === Online alpha / teacher-quality tracking (token-level) ===
        # These maintain streaming estimates for adaptive alpha adjustment
        self.alpha_ema: Optional[float] = None                 # EMA of α* estimate (for adaptive adjustment)
        self.teacher_quality_ema: Optional[float] = None       # EMA of teacher quality B²/A (internal)
        self.A_ema: Optional[float] = None                     # EMA of A(θ)=Var(X) (internal)
        self.B_ema: Optional[float] = None                     # EMA of B(θ)=Cov(Y,X) (internal)
        self.deltaV_ema: Optional[float] = None                # EMA of ΔV(α,θ) (internal)
        self.alpha_ema_beta: float = 0.9                       # EMA decay (window ~ 1/(1-β))
        self.last_alpha_stats: dict[str, float] = {}           # Stats to be logged to wandb

    
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
        
        for sample_idx, (seg_ids, info_rewards, info_stats, delta_phi) in info_results_by_index.items():
            boundaries = _compute_segment_boundaries(seg_ids)
            num_boundaries = len(boundaries)
            tool_use_count = info_stats.get("tool_use_count", 0)
            total_segments += num_boundaries
            
            print(f"Sample {sample_idx}: {num_boundaries} segments, {tool_use_count} tool uses, ΔΦ_total={info_stats.get('total_delta_phi', 0.0):.4f}")
            
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
            
            for sample_idx, (seg_ids, info_rewards, info_stats, delta_phi) in info_results_by_index.items():
                if info_stats.get("tool_use_count", 0) > 0:  # Check actual tool use count
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
        
        # Create stats tensor: [format_reward, f1, em, info_reward_total, tool_use_count, max_phi, min_phi, mean_phi, total_delta_phi, 
        #                      delta_phi_mean, delta_phi_std, delta_phi_max, delta_phi_min, delta_phi_abs_mean, 
        #                      delta_phi_positive_count, delta_phi_negative_count, delta_phi_zero_count,
        #                      delta_phi_q05, delta_phi_q25, delta_phi_median, delta_phi_q75, delta_phi_q95,
        #                      filtered_out, total_samples, final_phi]
        stats_tensor = torch.zeros((bs, 25), dtype=torch.float32)

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
                    if self.verbose:
                        print(f"[DEBUG] Sample {i}: No answer_candidates extracted from ground_truth, using default candidates")
                    sample_answer_candidates = self.answer_candidates
                
                info_reward_tasks.append((
                    i, prompt_str, response_str, sample_answer_candidates,
                    token_strings, self.vllm_model, self.info_timeout, self.verbose
                ))
        
        # Debug: log info reward task creation
        if self.enable_info_reward:
            print(f"[DEBUG] Created {len(info_reward_tasks)} info reward tasks out of {len(main_tasks)} total samples")

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
        if not effective_info_reward:
            print(f"[DEBUG] Info reward disabled: enable_info_reward={self.enable_info_reward}, step_weight={step_weight:.3f}, current_step={self.current_step}")
        elif not info_reward_tasks:
            print(f"[DEBUG] No info reward tasks created: enable_info_reward={self.enable_info_reward}, len(info_reward_tasks)=0")
        elif effective_info_reward and info_reward_tasks:
            print(f"Computing information rewards for {len(info_reward_tasks)} samples using {self.num_threads} threads...")
            
            # Split tasks into chunks for threading
            chunk_size = max(1, len(info_reward_tasks) // self.num_threads)
            chunks = [info_reward_tasks[i:i + chunk_size] for i in range(0, len(info_reward_tasks), chunk_size)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [
                    executor.submit(_process_info_reward_batch, chunk, self.vllm_api_base, self.info_timeout, 
                                  self.ppl_threshold)
                    for chunk in chunks
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_results = future.result()
                        info_results_by_index.update(chunk_results)
                    except Exception as e:
                        print(f"Error in information reward computation: {e}")
            
            # === SCALE APPLICATION ===
            # Apply scaling to raw delta_phi values (either immediate or fixed scale)
            if info_results_by_index:
                if self.enable_scale_control:
                    # === IMMEDIATE SCALE CONTROL ===
                    # Collect all raw delta_phi values to compute optimal scale for THIS batch
                    all_raw_delta_phi = []
                    for sample_idx, (seg_ids, info_rewards_unscaled, info_stats, delta_phi_raw) in info_results_by_index.items():
                        for raw_delta in delta_phi_raw:
                            if abs(raw_delta) > 1e-8:  # Only count non-zero values
                                all_raw_delta_phi.append(abs(raw_delta))
                    
                    if all_raw_delta_phi:
                        # Compute raw scale (absolute mean of raw delta_phi)
                        raw_scale = float(np.mean(all_raw_delta_phi))
                        
                        # Target scale: aim for the middle of the range
                        target_scale_mid = (self.target_scale_min + self.target_scale_max) / 2.0
                        
                        # Compute optimal scale to put current batch in target range
                        optimal_scale = target_scale_mid / max(raw_scale, 1e-8)
                        
                        # Apply immediate scale to all info_rewards in this batch
                        for sample_idx in info_results_by_index.keys():
                            seg_ids, info_rewards_unscaled, info_stats, delta_phi_raw = info_results_by_index[sample_idx]
                            # Scale the rewards immediately
                            info_rewards_scaled = [r * optimal_scale for r in info_rewards_unscaled]
                            delta_phi_scaled = [d * optimal_scale for d in delta_phi_raw]
                            
                            # Recompute stats based on scaled delta_phi
                            info_stats_scaled = info_stats.copy()
                            info_stats_scaled["final_phi_scaled"] = info_stats.get("final_phi", 0.0) * optimal_scale
                            
                            # Update delta_phi statistics to reflect SCALED values
                            if delta_phi_scaled:
                                delta_arr_scaled = np.array(delta_phi_scaled)
                                quantiles_scaled = np.percentile(delta_arr_scaled, [5, 25, 50, 75, 95])
                                
                                # Preserve raw stats for diagnostics (with _raw suffix)
                                info_stats_scaled["delta_phi_abs_mean_raw"] = info_stats.get("delta_phi_abs_mean", 0.0)
                                
                                # Replace with scaled stats
                                info_stats_scaled.update({
                                    "delta_phi_mean": float(np.mean(delta_arr_scaled)),
                                    "delta_phi_std": float(np.std(delta_arr_scaled)),
                                    "delta_phi_max": float(np.max(delta_arr_scaled)),
                                    "delta_phi_min": float(np.min(delta_arr_scaled)),
                                    "delta_phi_abs_mean": float(np.mean(np.abs(delta_arr_scaled))),  # SCALED abs mean
                                    "delta_phi_positive_count": int(np.sum(delta_arr_scaled > 0)),
                                    "delta_phi_negative_count": int(np.sum(delta_arr_scaled < 0)),
                                    "delta_phi_zero_count": int(np.sum(np.abs(delta_arr_scaled) < 1e-8)),
                                    "delta_phi_q05": float(quantiles_scaled[0]),
                                    "delta_phi_q25": float(quantiles_scaled[1]),
                                    "delta_phi_median": float(quantiles_scaled[2]),
                                    "delta_phi_q75": float(quantiles_scaled[3]),
                                    "delta_phi_q95": float(quantiles_scaled[4]),
                                })
                            
                            # Update with scaled values
                            info_results_by_index[sample_idx] = (seg_ids, info_rewards_scaled, info_stats_scaled, delta_phi_scaled)
                        
                        # Update delta_phi_scale for logging and next batch initialization
                        self.delta_phi_scale = optimal_scale
                        
                        # Compute actual scaled values for verification
                        actual_scaled_values = []
                        for sample_idx, (seg_ids, info_rewards_scaled, info_stats, delta_phi_scaled) in info_results_by_index.items():
                            for scaled_reward in info_rewards_scaled:
                                if abs(scaled_reward) > 1e-8:
                                    actual_scaled_values.append(abs(scaled_reward))
                        
                        batch_scale_after = float(np.mean(actual_scaled_values)) if actual_scaled_values else 0.0
                        
                        if self.verbose:
                            print(f"[immediate-scale-control] step={self.current_step}")
                            print(f"  raw_scale={raw_scale:.6f}, target_mid={target_scale_mid:.4f}")
                            print(f"  optimal_scale={optimal_scale:.6f} (raw × scale = {raw_scale * optimal_scale:.4f})")
                            print(f"  batch_scale_after={batch_scale_after:.4f} (should ≈ {target_scale_mid:.4f})")
                            print(f"  delta_phi_scale updated to {self.delta_phi_scale:.6f}")
                else:
                    # === FIXED SCALE (from init or adaptive alpha) ===
                    # Apply the current delta_phi_scale to all info_rewards
                    for sample_idx in info_results_by_index.keys():
                        seg_ids, info_rewards_unscaled, info_stats, delta_phi_raw = info_results_by_index[sample_idx]
                        # Scale with fixed delta_phi_scale
                        info_rewards_scaled = [r * self.delta_phi_scale for r in info_rewards_unscaled]
                        delta_phi_scaled = [d * self.delta_phi_scale for d in delta_phi_raw]
                        
                        # Recompute stats based on scaled delta_phi
                        info_stats_scaled = info_stats.copy()
                        info_stats_scaled["final_phi_scaled"] = info_stats.get("final_phi", 0.0) * self.delta_phi_scale
                        
                        # Update delta_phi statistics to reflect SCALED values
                        if delta_phi_scaled:
                            delta_arr_scaled = np.array(delta_phi_scaled)
                            quantiles_scaled = np.percentile(delta_arr_scaled, [5, 25, 50, 75, 95])
                            
                            # Preserve raw stats for diagnostics (with _raw suffix)
                            info_stats_scaled["delta_phi_abs_mean_raw"] = info_stats.get("delta_phi_abs_mean", 0.0)
                            
                            # Replace with scaled stats
                            info_stats_scaled.update({
                                "delta_phi_mean": float(np.mean(delta_arr_scaled)),
                                "delta_phi_std": float(np.std(delta_arr_scaled)),
                                "delta_phi_max": float(np.max(delta_arr_scaled)),
                                "delta_phi_min": float(np.min(delta_arr_scaled)),
                                "delta_phi_abs_mean": float(np.mean(np.abs(delta_arr_scaled))),  # SCALED abs mean
                                "delta_phi_positive_count": int(np.sum(delta_arr_scaled > 0)),
                                "delta_phi_negative_count": int(np.sum(delta_arr_scaled < 0)),
                                "delta_phi_zero_count": int(np.sum(np.abs(delta_arr_scaled) < 1e-8)),
                                "delta_phi_q05": float(quantiles_scaled[0]),
                                "delta_phi_q25": float(quantiles_scaled[1]),
                                "delta_phi_median": float(quantiles_scaled[2]),
                                "delta_phi_q75": float(quantiles_scaled[3]),
                                "delta_phi_q95": float(quantiles_scaled[4]),
                            })
                        
                        # Update with scaled values
                        info_results_by_index[sample_idx] = (seg_ids, info_rewards_scaled, info_stats_scaled, delta_phi_scaled)
                    
                    if self.verbose:
                        print(f"[fixed-scale] step={self.current_step}, delta_phi_scale={self.delta_phi_scale:.6f}")
            
            # Print segment-level info reward summary
            if info_results_by_index and self.verbose:
                self._print_segment_info_summary(info_results_by_index)
            
            # Save phi values to JSON file
            if info_results_by_index:
                _save_phi_values_json(info_results_by_index, self.current_step)
            
            # Save rollout data with phi values and info rewards
            if info_results_by_index and prompt_strs_list and response_strs_list:
                _save_rollout_data_with_rewards(
                    prompt_strs_list=prompt_strs_list,
                    response_strs_list=response_strs_list,
                    info_results_by_index=info_results_by_index,
                    main_score_by_index=main_score_by_index,
                    batch_metadata=batch_metadata,
                    step=self.current_step,
                    output_dir="rollout_data",
                    max_samples=None,  # Save all samples with info rewards
                    save_all=False  # Only save samples with info rewards
                )

        # Create mapping from sample index to token_strings for reward placement
        token_strings_by_index = {}
        for idx, metadata in enumerate(batch_metadata):
            sample_idx = metadata['index']
            if idx < len(batch_token_strings):
                token_strings_by_index[sample_idx] = batch_token_strings[idx]
        
        # Aggregate rewards into tensor
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())

            # Add information rewards per token to sequence positions
            adjusted_info_rewards = []  # Initialize for stats calculation
            final_phi_value_scaled = 0.0
            if i in info_results_by_index:
                seg_ids, info_rewards, info_stats, delta_phi = info_results_by_index[i]
                # Use scaled final_phi for PBRS consistency with scaled delta_phi
                final_phi_value_scaled = float(info_stats.get("final_phi_scaled", 0.0))
                if info_rewards and valid_response_length > 1:  # Need at least 2 tokens (one for info reward, one for final)
                    # Get response_mask for this sample
                    if "response_mask" in data_item.batch:
                        response_mask_tensor = data_item.batch["response_mask"][:valid_response_length]
                        response_mask = response_mask_tensor.cpu().tolist()
                        # print(f"response_mask found in data_item.batch")
                    else:
                        # print(f"response_mask not found in data_item.batch")
                        # Fallback: compute from attention_mask if response_mask not available
                        full_attention_mask = data_item.batch["attention_mask"]
                        response_attention_mask = full_attention_mask[prompt_length:prompt_length + valid_response_length]
                        response_mask = response_attention_mask.cpu().tolist()
                    
                    # Get token_strings for this sample
                    sample_token_strings = token_strings_by_index.get(i, None)
                    
                    # Place rewards at <|im_end|> after </tool_call> in each segment
                    adjusted_info_rewards = _per_token_info_rewards_with_mask(
                        seg_ids, delta_phi, response_mask, sample_token_strings
                    )
                    
                    max_pos = min(len(adjusted_info_rewards), valid_response_length - 1)  # Avoid overwriting final reward
                    for pos in range(max_pos):
                        # Apply both configured weight and step-based weight
                        effective_weight = self.info_reward_weight * step_weight
                        reward_tensor[i, pos] += float(adjusted_info_rewards[pos]) * effective_weight
                
                # Update stats tensor with info reward stats
                stats_tensor[i, 4] = float(info_stats.get("tool_use_count", 0))  # Actual tool use count (excludes initial boundary)
                stats_tensor[i, 5] = float(info_stats.get("max_phi", 0.0))
                stats_tensor[i, 6] = float(info_stats.get("min_phi", 0.0))
                stats_tensor[i, 7] = float(info_stats.get("mean_phi", 0.0))  # Mean of all phi values
                # total_delta_phi includes the final correction: sum(ΔΦ) - Φ_final
                # This ensures PBRS property: total shaped reward = sum(ΔΦ) - Φ_final (already scaled)
                stats_tensor[i, 8] = float(info_stats.get("total_delta_phi", 0.0)) - float(final_phi_value_scaled)
                
                # Add delta_phi scale statistics
                stats_tensor[i, 9] = float(info_stats.get("delta_phi_mean", 0.0))
                stats_tensor[i, 10] = float(info_stats.get("delta_phi_std", 0.0))
                stats_tensor[i, 11] = float(info_stats.get("delta_phi_max", 0.0))
                stats_tensor[i, 12] = float(info_stats.get("delta_phi_min", 0.0))
                stats_tensor[i, 13] = float(info_stats.get("delta_phi_abs_mean", 0.0))
                stats_tensor[i, 14] = float(info_stats.get("delta_phi_positive_count", 0))
                stats_tensor[i, 15] = float(info_stats.get("delta_phi_negative_count", 0))
                stats_tensor[i, 16] = float(info_stats.get("delta_phi_zero_count", 0))
                stats_tensor[i, 17] = float(info_stats.get("delta_phi_q05", 0.0))
                stats_tensor[i, 18] = float(info_stats.get("delta_phi_q25", 0.0))
                stats_tensor[i, 19] = float(info_stats.get("delta_phi_median", 0.0))
                stats_tensor[i, 20] = float(info_stats.get("delta_phi_q75", 0.0))
                stats_tensor[i, 21] = float(info_stats.get("delta_phi_q95", 0.0))
                stats_tensor[i, 22] = float(1.0 if info_stats.get("filtered_out", False) else 0.0)  # filtered_out flag
                stats_tensor[i, 23] = 1.0  # total_samples counter
                stats_tensor[i, 24] = final_phi_value_scaled  # Store scaled final_phi

            # Main score goes to the final valid position (like naive.py)
            main_score = main_score_by_index.get(i, 0.0)
            main_extra = main_extra_by_index.get(i, {})
            main_stats = main_stats_by_index.get(i, {})
            
            # Final reward goes to the last valid token position (like naive.py)
            # Apply PBRS: r'(s_T) = r(s_T) + γΦ(terminal) - Φ(s_T)
            # Assuming Φ(terminal) = 0, we get: r'(s_T) = r(s_T) - Φ(s_T)
            if valid_response_length > 0:
                # Apply both configured weight and step-based weight to final phi
                # Note: final_phi_value_scaled is already scaled (by optimal_scale or delta_phi_scale)
                effective_weight = self.info_reward_weight * step_weight
                final_phi_correction = -final_phi_value_scaled * effective_weight
                reward_tensor[i, valid_response_length - 1] += float(main_score) + final_phi_correction
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
            # info_reward_total is sum(ΔΦ) - Φ_final (already scaled) to maintain PBRS consistency
            info_reward_total_scaled = float(sum(adjusted_info_rewards) if adjusted_info_rewards else 0.0)
            info_reward_total_scaled += (-final_phi_value_scaled)  # Add final correction (already scaled)
            stats_tensor[i, 3] = info_reward_total_scaled
            
            # Set total_samples counter for all samples (including those without info rewards)
            stats_tensor[i, 23] = 1.0

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
        # if True:
        #     try:
        #         # Create mapping from sample index to strings
        #         prompt_strs_dict = {}
        #         response_strs_dict = {}
        #         for idx, metadata in enumerate(batch_metadata):
        #             i = metadata['index']
        #             prompt_strs_dict[i] = prompt_strs_list[idx]
        #             response_strs_dict[i] = response_strs_list[idx]
                
        #         _save_sample_data_info_reward(
        #             data, reward_tensor, main_extra_by_index, info_results_by_index, 
        #             prompt_strs_dict, response_strs_dict, self.tokenizer, max_samples=1
        #         )
        #     except Exception as e:
        #         print(f"Warning: Failed to save sample data: {e}")
                # === Trajectory-level variance / covariance stats for alpha & teacher quality ===
        # Compute episode-level statistics where each trajectory contributes one sample:
        #   Y_i = main_score for trajectory i (environment return)
        #   X_i = φ_final - φ_0 for trajectory i (total teacher signal)
        # 
        # This gives unbiased batch-level estimates of:
        #   A_hat = Var(X), B_hat = Cov(Y,X), C_hat = Var(Y),
        #   α* = B/A (variance-minimizing scale),
        #   ΔV_hat(α,θ) = A α² - 2B α (variance change),
        #   TeacherQuality = B² / A
        #
        # Key improvements over token-level sampling:
        #   1. Matches theoretical definition (episode-level random variables)
        #   2. No length-based weighting bias (equal weight per trajectory)
        #   3. Independent samples (no correlation from repeated tokens)
        #   4. Stable A/B estimates (effective sample size = number of trajectories)
        if info_results_by_index:
            all_X: list[float] = []
            all_Y: list[float] = []

            for idx, metadata in enumerate(batch_metadata):
                sample_idx = metadata["index"]
                if sample_idx not in info_results_by_index:
                    continue

                main_score = float(main_score_by_index.get(sample_idx, 0.0))
                seg_ids, _info_rewards, info_stats, _delta_phi = info_results_by_index[sample_idx]
                phi_values = info_stats.get("phi_values", [])

                # We need at least one boundary Φ_i besides the initial Φ_0
                if not phi_values or len(phi_values) < 2:
                    continue

                # Trajectory-level statistics: one sample per trajectory
                # Y_i = environment return (main_score)
                Y_i = main_score
                
                # X_i = total teacher signal (φ_final - φ_0)
                # Using relative phi removes prompt difficulty/length bias
                phi_0 = float(phi_values[0])      # Initial potential (before any tool use)
                phi_final = float(phi_values[-1])  # Final potential (after all tool use)
                X_i = phi_final - phi_0
                
                # Each trajectory contributes exactly one sample
                all_X.append(X_i)
                all_Y.append(Y_i)

            if len(all_X) >= 2:
                X_arr = np.asarray(all_X, dtype=np.float64)
                Y_arr = np.asarray(all_Y, dtype=np.float64)

                # Unbiased sample estimates on the current batch
                var_Y = float(np.var(Y_arr, ddof=1))  # C ≈ Var(Y)
                var_X = float(np.var(X_arr, ddof=1))  # A_hat(θ) ≈ Var(X)
                if var_X > 0.0:
                    cov_YX = float(np.cov(Y_arr, X_arr, ddof=1)[0, 1])  # B_hat(θ) ≈ Cov(Y,X)
                    alpha_hat = cov_YX / var_X
                    teacher_quality = (cov_YX * cov_YX) / var_X
                else:
                    cov_YX = 0.0
                    alpha_hat = 0.0
                    teacher_quality = 0.0

                # Effective α actually used in shaping for this step
                # (theoretical α is delta_phi_scale; runtime还乘了info_reward_weight和step_weight)
                alpha_effective = (
                    self.delta_phi_scale
                    * self.info_reward_weight
                    * self._compute_info_reward_step_weight(self.current_step)
                )

                # ΔV_hat(α,θ) = A_hat α^2 - 2 B_hat α  —— 是否 still variance-reducing
                deltaV_batch = var_X * (alpha_effective ** 2) - 2.0 * cov_YX * alpha_effective

                # Teacher correlation R^2 ≈ B^2 / (Var(Y) Var(X))
                eps = 1e-8
                if var_Y > eps and var_X > eps:
                    teacher_r2 = (cov_YX * cov_YX) / (var_Y * var_X)
                else:
                    teacher_r2 = 0.0

                # Ratio of std of shaped component vs outcome component
                std_teacher = abs(alpha_effective) * float(np.sqrt(max(var_X, 0.0)))
                std_env = float(np.sqrt(max(var_Y, 0.0)))
                if std_env > eps:
                    std_ratio = std_teacher / std_env
                else:
                    std_ratio = 0.0

                # EMA over steps (streaming estimates A_t, B_t, ΔV_t, TeacherQuality_t)
                beta = self.alpha_ema_beta
                if self.alpha_ema is None:
                    # first step: initialize EMAs with batch stats
                    self.alpha_ema = alpha_hat
                    self.teacher_quality_ema = teacher_quality
                    self.A_ema = var_X
                    self.B_ema = cov_YX
                    self.deltaV_ema = deltaV_batch
                else:
                    self.alpha_ema = beta * self.alpha_ema + (1.0 - beta) * alpha_hat
                    self.teacher_quality_ema = (
                        beta * self.teacher_quality_ema + (1.0 - beta) * teacher_quality
                    )
                    self.A_ema = beta * self.A_ema + (1.0 - beta) * var_X
                    self.B_ema = beta * self.B_ema + (1.0 - beta) * cov_YX
                    self.deltaV_ema = beta * self.deltaV_ema + (1.0 - beta) * deltaV_batch

                # Note: Adaptive alpha is disabled when immediate scale control is enabled
                # because immediate scale control directly sets delta_phi_scale each batch
                if self.enable_adaptive_alpha and not self.enable_scale_control and self.alpha_ema is not None:
                    # 想让 alpha_effective ≈ alpha_ema
                    denom = self.info_reward_weight * self._compute_info_reward_step_weight(self.current_step)
                    denom = max(denom, 1e-8)
                    target_scale = self.alpha_ema / denom   # 注意下面会再 clamp
                    
                    # 用 EMA 慢慢跟过去
                    self.delta_phi_scale = (
                        (1.0 - self.alpha_update_rate) * self.delta_phi_scale
                        + self.alpha_update_rate * target_scale
                    )
                    self.delta_phi_scale = max(self.alpha_min, min(self.alpha_max, self.delta_phi_scale))

        # === Collect scale stats for logging ===
        # Note: immediate scale control has already been applied above
        scale_stats = {}
        if self.enable_scale_control and info_results_by_index:
            # Collect actual scaled info rewards for logging
            all_scaled_info_rewards = []
            for sample_idx, (seg_ids, info_rewards, info_stats, delta_phi) in info_results_by_index.items():
                for reward_val in info_rewards:
                    if abs(reward_val) > 1e-8:
                        all_scaled_info_rewards.append(abs(reward_val))
            
            if all_scaled_info_rewards:
                batch_scale = float(np.mean(all_scaled_info_rewards))
                scale_stats = {
                    "info_reward_scale_batch": batch_scale,
                    "target_scale_min": self.target_scale_min,
                    "target_scale_max": self.target_scale_max,
                }

        # Cache stats for external logging / schedulers (only if we have trajectory-level stats)
        if info_results_by_index:
            # Check if we have trajectory-level alpha stats from earlier
            if not hasattr(self, 'last_alpha_stats'):
                self.last_alpha_stats = {}
            
            # Add scale control stats if available
            if scale_stats:
                self.last_alpha_stats.update(scale_stats)
            
            # Always update delta_phi_scale in stats
            self.last_alpha_stats["delta_phi_scale"] = self.delta_phi_scale
            
            # Note: verbose logging is already done in the scale control section above
            # No need for additional logging here
        
        # Print verbose info for alpha tracking (if available)
        if hasattr(self, 'last_alpha_stats') and 'alpha_hat_batch' in self.last_alpha_stats:
            if self.verbose:
                print(
                    f"[teacher-quality] step={self.current_step} "
                    f"α*_batch={self.last_alpha_stats['alpha_hat_batch']:.4f} "
                    f"α_ema={self.last_alpha_stats.get('alpha_ema', 0.0):.4f} "
                    f"α_eff={self.last_alpha_stats.get('alpha_effective', 0.0):.4f} "
                    f"ΔV={self.last_alpha_stats.get('deltaV_batch', 0.0):.4f} "
                    f"R²={self.last_alpha_stats.get('teacher_r2_batch', 0.0):.4f} "
                    f"scale={self.delta_phi_scale:.6f} "
                    f"n_traj={int(self.last_alpha_stats.get('num_trajectory_samples', 0))}"
                )

        # Add alpha tracking stats to reward_extra_info for logging
        # These are batch-level stats (same value for all samples in the batch)
        if self.last_alpha_stats:
            for key, value in self.last_alpha_stats.items():
                reward_extra_info[key] = [value] * len(data)  # Repeat for each sample in batch
        else:
            # Initialize empty stats if alpha tracking hasn't run yet
            default_alpha_stats = {
                # Core batch-level stats
                "alpha_hat_batch": 0.0,
                "deltaV_batch": 0.0,
                "teacher_r2_batch": 0.0,
                "num_trajectory_samples": 0.0,
                # Raw statistics
                "A_batch": 0.0,
                "B_batch": 0.0,
                "C_batch": 0.0,
                # EMA smoothed alpha
                "alpha_ema": 0.0,
                # Effective alpha
                "alpha_effective": 0.0,
                "delta_phi_scale": self.delta_phi_scale,
            }
            for key, value in default_alpha_stats.items():
                reward_extra_info[key] = [value] * len(data)
        
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
