#!/usr/bin/env bash
set -euo pipefail

# Privacy-safe PPO example launcher.
# - No hardcoded secrets
# - No private absolute paths
# - Easy reward manager switching
#
# Usage:
#   chmod +x examples/release/run_search_ppo_example.sh
#   TRAIN_DATA=/path/train.parquet VAL_DATA=/path/val.parquet \
#   TOOL_CONFIG=/path/tool_config.yaml \
#   bash examples/release/run_search_ppo_example.sh
#
# Optional reward manager switch:
#   REWARD_MANAGER=naive
#   REWARD_MANAGER=naive_llm
#   REWARD_MANAGER=execution_reward
#   REWARD_MANAGER=info_reward_llm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_BIN_PATH="${CONDA_BIN_PATH:-}"
if [[ -z "${CONDA_BIN_PATH}" ]]; then
  echo "Please set CONDA_BIN_PATH, e.g. /opt/conda/envs/verl/bin"
  exit 1
fi

PYTHON_BIN="${CONDA_BIN_PATH%/}/python"
RAY_BIN="${CONDA_BIN_PATH%/}/ray"

NUM_GPUS="${NUM_GPUS:-8}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/examples/sglang_multiturn/config}"
CONFIG_NAME="${CONFIG_NAME:-search_multiturn_grpo}"
TOOL_CONFIG="${TOOL_CONFIG:-${CONFIG_PATH}/tool_config/search_tool_config.yaml}"

TRAIN_DATA="${TRAIN_DATA:-}"
VAL_DATA="${VAL_DATA:-}"
if [[ -z "${TRAIN_DATA}" || -z "${VAL_DATA}" ]]; then
  echo "Please set TRAIN_DATA and VAL_DATA."
  exit 1
fi

# Logging (no secrets in repo)
WANDB_PROJECT="${WANDB_PROJECT:-search_r1_like_async_rl}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_EXPERIMENT_NAME="${WANDB_EXPERIMENT_NAME:-qwen7b-ppo-example}"
if [[ -n "${WANDB_ENTITY}" ]]; then
  export WANDB_ENTITY
fi
export WANDB_PROJECT

# Reward settings
REWARD_MANAGER="${REWARD_MANAGER:-naive}"
REWARD_SCORE_SOURCE="${REWARD_SCORE_SOURCE:-em}"

build_reward_args() {
  case "${REWARD_MANAGER}" in
    naive)
      echo "reward_model.reward_manager=naive"
      ;;
    naive_llm)
      cat <<EOF
reward_model.reward_manager=naive_llm
reward_model.reward_kwargs.enable_execution_reward=true
reward_model.reward_kwargs.enable_process_evaluation=false
reward_model.reward_kwargs.score_source=${REWARD_SCORE_SOURCE}
EOF
      ;;
    execution_reward)
      cat <<EOF
reward_model.reward_manager=execution_reward
reward_model.reward_kwargs.enable_execution_reward=true
reward_model.reward_kwargs.reward_distribution_mode=last_token
reward_model.reward_kwargs.execution_reward_weight=1.0
reward_model.reward_kwargs.score_source=${REWARD_SCORE_SOURCE}
EOF
      ;;
    info_reward_llm)
      cat <<EOF
reward_model.reward_manager=info_reward_llm
reward_model.reward_kwargs.enable_info_reward=true
reward_model.reward_kwargs.info_reward_weight=1.0
reward_model.reward_kwargs.delta_phi_scale=0.02
reward_model.reward_kwargs.score_source=${REWARD_SCORE_SOURCE}
EOF
      ;;
    *)
      echo "Unsupported REWARD_MANAGER=${REWARD_MANAGER}" >&2
      exit 1
      ;;
  esac
}

mapfile -t REWARD_ARGS < <(build_reward_args)

export NCCL_DEBUG="${NCCL_DEBUG:-info}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

mkdir -p "${PROJECT_DIR}/slurm"
mkdir -p "${PROJECT_DIR}/rollout_data/${WANDB_EXPERIMENT_NAME}"

cd "${PROJECT_DIR}"

"${RAY_BIN}" stop -f || true
"${RAY_BIN}" start --head --num-gpus "${NUM_GPUS}" --include-dashboard=True --dashboard-port 8265
sleep 3

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  --config-path="${CONFIG_PATH}" \
  --config-name="${CONFIG_NAME}" \
  algorithm.adv_estimator=gae \
  data.train_batch_size=256 \
  data.val_batch_size=256 \
  data.max_prompt_length=4096 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  data.return_raw_chat=true \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.max_model_len=15000 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  critic.ppo_micro_batch_size_per_gpu=8 \
  critic.model.path="${BASE_MODEL}" \
  algorithm.use_kl_in_reward=false \
  trainer.critic_warmup=0 \
  trainer.val_before_train=false \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=500 \
  trainer.rollout_data_dir="./rollout_data/${WANDB_EXPERIMENT_NAME}" \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  trainer.log_val_generations=50 \
  trainer.total_epochs=2 \
  "${REWARD_ARGS[@]}"

"${RAY_BIN}" stop -f

