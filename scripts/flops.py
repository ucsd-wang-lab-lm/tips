# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
# ]
# ///
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class QwenLikeConfig:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    head_dim: Optional[int] = None  # if None, assume hidden_size // num_attention_heads


def qwen2_param_constants(cfg: QwenLikeConfig) -> Dict[str, int]:
    """
    Reproduce VERL-style dense_N and attention constants for Qwen/LLaMA-like blocks.
    """
    hidden_size = cfg.hidden_size
    vocab_size = cfg.vocab_size
    num_hidden_layers = cfg.num_hidden_layers
    num_key_value_heads = cfg.num_key_value_heads
    num_attention_heads = cfg.num_attention_heads
    intermediate_size = cfg.intermediate_size

    head_dim = cfg.head_dim or (hidden_size // num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # SwiGLU MLP with up/down + gated, and separate Q/K/V/O proj in attention
    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (
        q_size + k_size + v_size + num_attention_heads * head_dim
    )
    emd_and_lm_head_N = vocab_size * hidden_size * 2

    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N

    return dict(
        dense_N=dense_N,
        head_dim=head_dim,
        num_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
    )


def fwd_flops(cfg: QwenLikeConfig, batch_seqlens: List[float]) -> Dict[str, float]:
    """
    FLOPs of a single forward pass (no gradients) for arbitrary batch_seqlens.
    Treats each sequence as an independent full-context pass (no KV reuse).
    """
    consts = qwen2_param_constants(cfg)
    dense_N = consts["dense_N"]
    head_dim = consts["head_dim"]
    num_layers = consts["num_layers"]
    num_attention_heads = consts["num_attention_heads"]

    tokens_sum = sum(batch_seqlens)
    seqlen_square_sum = sum(s * s for s in batch_seqlens)

    # VERL: fwd+bwd dense ≈ 6 * dense_N * tokens; use ~1/3 for fwd-only
    dense_flops = 2.0 * dense_N * tokens_sum

    # VERL: fwd+bwd attn ≈ 12 * sum(L^2) * head_dim * H * L; fwd-only ~1/3
    attn_flops = 4.0 * seqlen_square_sum * head_dim * num_attention_heads * num_layers

    total_flops = dense_flops + attn_flops

    return dict(dense=dense_flops, attn=attn_flops, total=total_flops)


def fwd_flops_with_prefix_cache_for_segments(
    cfg: QwenLikeConfig,
    *,
    batch_size: int,
    segment_lengths: List[float],
    answer_avg_len: float = 0.0,
    avg_num_answers: float = 1.0,
) -> Dict[str, float]:
    """
    Approximate FLOPs for:
      - A set of monotone-increasing, prefix-sharing segments (prefix_i),
      - Plus scoring answer continuations of avg length 'answer_avg_len'
        appended after each prefix,
      - For avg_num_answers candidate answers per prefix.

    Assumptions:
      - segment_lengths[i] are lengths of prefixes for a single conversation,
        and they form a strictly increasing chain (full prefix sharing).
      - We reuse KV cache across prefixes, so we only pay for a single
        forward of length L_max over the prefixes.
      - For each prefix_i and each answer candidate, we decode 'answer_avg_len'
        tokens with KV cache (context length = prefix_i + previous answer tokens).
    """
    consts = qwen2_param_constants(cfg)
    dense_N = consts["dense_N"]
    head_dim = consts["head_dim"]
    num_layers = consts["num_layers"]
    num_attention_heads = consts["num_attention_heads"]

    B = float(batch_size)
    L_max = float(max(segment_lengths)) if segment_lengths else 0.0

    # ---------- Prefix compute (shared KV) ----------
    tokens_prefix = B * L_max
    seqlen_square_sum_prefix = B * (L_max * L_max)

    dense_prefix_flops = 2.0 * dense_N * tokens_prefix
    attn_prefix_flops = (
        4.0 * seqlen_square_sum_prefix * head_dim * num_attention_heads * num_layers
    )

    # ---------- Answer decode compute ----------
    La = float(answer_avg_len)
    num_answers = float(avg_num_answers)
    S = float(len(segment_lengths))

    dense_answers_flops = 0.0
    attn_answers_flops = 0.0

    if La > 0.0 and S > 0.0 and num_answers > 0.0:
        # Total answer tokens:
        #   tokens_answer = B * S * La * num_answers
        tokens_answer = B * S * La * num_answers
        dense_answers_flops = 2.0 * dense_N * tokens_answer

        # Attention: decode with KV cache.
        # For prefix length Lp and answer length La:
        #   sum_t context_len_t = La * Lp + La*(La - 1)/2
        # Sum this over all prefixes, all answers, all batch elements.
        context_sum_answers = 0.0
        for Lp in segment_lengths:
            Lp = float(Lp)
            context_sum_answers += La * Lp + La * (La - 1.0) / 2.0

        # Multiply by batch size and number of answers
        context_sum_answers *= B * num_answers

        attn_answers_flops = (
            4.0 * context_sum_answers * head_dim * num_attention_heads * num_layers
        )

    dense_total = dense_prefix_flops + dense_answers_flops
    attn_total = attn_prefix_flops + attn_answers_flops
    total_flops = dense_total + attn_total

    return dict(
        dense_prefix=dense_prefix_flops,
        attn_prefix=attn_prefix_flops,
        dense_answers=dense_answers_flops,
        attn_answers=attn_answers_flops,
        dense=dense_total,
        attn=attn_total,
        total=total_flops,
    )


def estimate_grpo_step_flops(
    *,
    cfg: QwenLikeConfig,
    batch_size: int,
    group_size: int,
    avg_prompt_len: int,
    avg_gen_len: int,
    policy_passes_per_step: int = 1,
    use_critic: bool = False,
    critic_passes_per_step: Optional[int] = None,
) -> Dict[str, float]:
    """
    FLOPs for ONE GRPO optimizer step:

      1) Generation (prefill + decode with KV cache), forward-only.
      2) Policy update: policy_passes_per_step * (fwd + bwd) on sequences
         of length (prompt_len + gen_len).
      3) Optional critic update: critic_passes_per_step * (fwd + bwd) on
         same sequences, same batch_size and group_size.

    critic_passes_per_step defaults to policy_passes_per_step if use_critic
    is True and critic_passes_per_step is None.
    """
    consts = qwen2_param_constants(cfg)
    dense_N = consts["dense_N"]
    head_dim = consts["head_dim"]
    num_layers = consts["num_layers"]
    num_attention_heads = consts["num_attention_heads"]

    B = batch_size
    G = group_size
    Lp = float(avg_prompt_len)
    Lg = float(avg_gen_len)

    num_seqs = B * G  # total rollouts in this step
    S_total = Lp + Lg

    # ---------- 1. Generation phase (forward-only) ----------

    tokens_gen = num_seqs * (Lp + Lg)

    # Dense: fwd-only
    generation_dense_flops = 2.0 * dense_N * tokens_gen

    # Attention: prefill (full-context) + decode (KV cache)
    # Prefill
    seqlen_square_sum_prefill = num_seqs * (Lp**2)
    attn_prefill_flops = (
        4.0 * seqlen_square_sum_prefill * head_dim * num_attention_heads * num_layers
    )

    # Decode with KV cache:
    # per sequence: sum_t context_len_t = Lg*Lp + Lg*(Lg-1)/2
    context_sum_decode = Lg * Lp + Lg * (Lg - 1.0) / 2.0
    attn_decode_flops = (
        4.0
        * num_seqs
        * context_sum_decode
        * head_dim
        * num_attention_heads
        * num_layers
    )

    generation_attn_flops = attn_prefill_flops + attn_decode_flops
    generation_total_flops = generation_dense_flops + generation_attn_flops

    # ---------- 2. Policy update (fwd+bwd over this batch) ----------

    tokens_train = num_seqs * S_total

    # Per-pass (one fwd+bwd) cost
    policy_dense_per_pass = 6.0 * dense_N * tokens_train
    seqlen_square_sum_train = num_seqs * (S_total**2)
    policy_attn_per_pass = (
        12.0 * seqlen_square_sum_train * head_dim * num_attention_heads * num_layers
    )

    # Scale by policy_passes_per_step
    policy_dense_flops = policy_dense_per_pass * policy_passes_per_step
    policy_attn_flops = policy_attn_per_pass * policy_passes_per_step
    policy_total_flops = policy_dense_flops + policy_attn_flops

    # ---------- 3. Critic update (optional, same config as policy) ----------

    critic_dense_flops = 0.0
    critic_attn_flops = 0.0
    critic_total_flops = 0.0

    if use_critic:
        if critic_passes_per_step is None:
            critic_passes_per_step = policy_passes_per_step

        critic_dense_flops = policy_dense_per_pass * critic_passes_per_step
        critic_attn_flops = policy_attn_per_pass * critic_passes_per_step
        critic_total_flops = critic_dense_flops + critic_attn_flops

    # ---------- Total per step ----------

    step_total_flops = generation_total_flops + policy_total_flops + critic_total_flops

    return dict(
        generation_dense=generation_dense_flops,
        generation_attn=generation_attn_flops,
        generation_total=generation_total_flops,
        policy_dense=policy_dense_flops,
        policy_attn=policy_attn_flops,
        policy_total=policy_total_flops,
        critic_dense=critic_dense_flops,
        critic_attn=critic_attn_flops,
        critic_total=critic_total_flops,
        step_total=step_total_flops,
    )


def estimate_grpo_total_flops(
    *,
    cfg: QwenLikeConfig,
    batch_size: int,
    group_size: int,
    avg_prompt_len: int,
    avg_gen_len: int,
    num_steps: int,
    policy_passes_per_step: int = 1,
    use_critic: bool = False,
    critic_passes_per_step: Optional[int] = None,
) -> Dict[str, float]:
    """
    Total FLOPs over num_steps, with optional critic.

    num_steps is the number of optimizer steps (GRPO updates).
    """
    step_stats = estimate_grpo_step_flops(
        cfg=cfg,
        batch_size=batch_size,
        group_size=group_size,
        avg_prompt_len=avg_prompt_len,
        avg_gen_len=avg_gen_len,
        policy_passes_per_step=policy_passes_per_step,
        use_critic=use_critic,
        critic_passes_per_step=critic_passes_per_step,
    )

    total = {k: v * num_steps for k, v in step_stats.items()}
    total["total_training_flops"] = total["step_total"]
    return total


# Example usage
if __name__ == "__main__":
    qwen25_7b_cfg = QwenLikeConfig(
        hidden_size=3584,
        vocab_size=152064,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        intermediate_size=18944,
        head_dim=128,  # 3584 // 28
    )
    qwen25_3b_cfg = QwenLikeConfig(
        hidden_size=2048,
        vocab_size=151936,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
        intermediate_size=11008,
        head_dim=128,  # 2048 // 16
    )
    current_config = qwen25_7b_cfg
    batch_size = 256
    avg_prompt_len = 400
    avg_gen_len = 4096

    tips_grpo_config = {
        "batch_size": batch_size,
        "avg_prompt_len": avg_prompt_len,
        "avg_gen_len": avg_gen_len,
        "group_size": 8,
        "use_critic": False,
    }
    tips_ppo_config = {
        "batch_size": batch_size,
        "avg_prompt_len": avg_prompt_len,
        "avg_gen_len": avg_gen_len,
        "group_size": 1,
        "use_critic": True,
    }

    n_tool_calls = 5
    tool_call_lengths = []
    for n in range(n_tool_calls):
        tool_call_lengths.append(avg_prompt_len + avg_gen_len / n_tool_calls * n)

    # Teacher scoring parameters
    answer_avg_len = 10.0  # average tokens per candidate answer A
    avg_num_answers = 2.0  # average number of candidate answers per prefix

    # Teacher scoring: prefixes share KV; answers are decoded on top of each prefix,
    # for avg_num_answers candidates.
    teacher_scoring_flops = fwd_flops_with_prefix_cache_for_segments(
        current_config,
        batch_size=batch_size,
        segment_lengths=tool_call_lengths,
        answer_avg_len=answer_avg_len,
        avg_num_answers=avg_num_answers,
    )
    print(
        f"Teacher scoring FLOPs (with KV sharing + answers): "
        f"{teacher_scoring_flops['total'] / 1e12:.3f} TFLOPs"
    )

    tips_grpo_step = estimate_grpo_step_flops(cfg=current_config, **tips_grpo_config)
    tips_ppo_step = estimate_grpo_step_flops(cfg=current_config, **tips_ppo_config)
    print("\nFLOPs per GRPO step:")
    print(f"{tips_grpo_step['step_total'] / 1e12:.3f} TFLOPs")
    print("\nFLOPs per PPO step:")
    print(f"{tips_ppo_step['step_total'] / 1e12:.3f} TFLOPs")

    print("Percent increase in PPO flops with teacher scoring:")
    print(
        f"{100 * (tips_ppo_step['step_total'] + teacher_scoring_flops['total']) / tips_ppo_step['step_total'] - 100:.3f}%"
    )
