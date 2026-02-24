# TIPS: Turn-level Information-Potential Reward Shaping for Search-Augmented LLMs.

This repository contains the codebase for TIPS, a `verl`-based project for Search-R1 style RL training with tool use (multi-turn retrieval, PPO/GRPO training, and validation workflows).

ICLR 2026 poster: [https://openreview.net/pdf?id=eBMOr6a84z](https://openreview.net/pdf?id=eBMOr6a84z)

This README is written as a release-oriented guide:
- install requirements
- run retrieval + training
- switch reward managers and key params
- understand core code paths for experiments

## Privacy and release notes

- Do not commit secrets (for example `WANDB_API_KEY`).
- Do not commit private filesystem paths (for example `/mnt/weka/...`).
- Use environment variables for all cluster-specific settings.
- Release-ready examples are in `examples/release/`.

## Requirements

- Python: `3.10` to `3.12`
- Package manager: [uv](https://docs.astral.sh/uv/) (`>=0.5`)
- GPU training: Linux + CUDA toolchain (CUDA 12.1 recommended)
- Optional: Ray cluster (single node is supported and used by the example scripts)

## Installation

### 1) Create environment and install base dependencies

```bash
uv sync --python 3.10
source .venv/bin/activate
```

On Linux with CUDA 12.1 wheels:

```bash
UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121" uv sync
```

### 2) Optional extras

```bash
# Retriever stack
uv sync --extra retriever

# Logging integrations
uv sync --extra logging

# Evaluation helpers
uv sync --extra evaluation
```

You can combine extras, for example:

```bash
uv sync --extra retriever --extra logging --extra evaluation
```

## Data preprocessing

The repository includes a preprocessing pipeline for Search-R1 style data under `examples/data_preprocess/`.

### 1) Download and convert dataset format

Use `preprocess_search_r1_dataset.py` to download train/test parquet from a Hugging Face dataset repo and convert them to the training format expected by multi-turn RL:

```bash
python examples/data_preprocess/preprocess_search_r1_dataset.py \
  --hf_repo_id PeterJinGo/nq_hotpotqa_train \
  --local_dir ./searchR1_processed_direct
```

This step produces:
- `./searchR1_processed_direct/train.parquet`
- `./searchR1_processed_direct/test.parquet`

### 2) Filter by token length and sample by data source

Use `analyze_search_r1_dataset.py` to:
- compute prompt token lengths with your tokenizer
- filter by min/max token length
- uniformly sample each data source
- write final train/test parquet files for training

```bash
python examples/data_preprocess/analyze_search_r1_dataset.py \
  --data_dir ./searchR1_processed_direct \
  --tokenizer_name Qwen/Qwen2.5-7B-Instruct \
  --max_tokens 4000 \
  --sampling_ratio 0.125 \
  --output_dir ./data_preprocess/searchR1_processed
```

This step produces:
- `./data_preprocess/searchR1_processed/train_processed.parquet`
- `./data_preprocess/searchR1_processed/test_processed.parquet`

### 3) Analyze target-answer distribution (optional)

`run_target_analysis.sh` is a convenience wrapper for `analyze_target_answers.py`:

```bash
bash examples/data_preprocess/run_target_analysis.sh \
  ./data_preprocess/searchR1_processed \
  ./target_answers_analysis_results
```

### 4) Connect processed data to training scripts

Set training file paths to the processed parquet files:

```bash
export TRAIN_DATA=./data_preprocess/searchR1_processed/train_processed.parquet
export VAL_DATA=./data_preprocess/searchR1_processed/test_processed.parquet
```

## Quick start (release example scripts)

Two release-safe launchers are provided:

- PPO example: `examples/release/run_search_ppo_example.sh`
- GRPO example: `examples/release/run_search_grpo_example.sh`

Both scripts:
- avoid hardcoded secrets and private paths
- support reward manager switching via env vars
- use `verl.trainer.main_ppo`

### Required environment variables

```bash
export CONDA_BIN_PATH=/path/to/conda_env/bin
export TRAIN_DATA=/path/to/train.parquet
export VAL_DATA=/path/to/val.parquet
export TOOL_CONFIG=/path/to/search_tool_config.yaml
```

Optional logging variables:

```bash
export WANDB_PROJECT=search_r1_like_async_rl
export WANDB_ENTITY=your_wandb_entity
export WANDB_EXPERIMENT_NAME=my_run_name
# export WANDB_API_KEY=...   # set in shell or CI secret, never commit
```

### Run PPO example

```bash
REWARD_MANAGER=naive \
bash examples/release/run_search_ppo_example.sh
```

### Run GRPO example

```bash
ADV_ESTIMATOR=grpo \
REWARD_MANAGER=naive \
bash examples/release/run_search_grpo_example.sh
```

Use `ADV_ESTIMATOR=sgrpo` to run SGRPO.

## Reward manager switching

The release scripts support:

- `REWARD_MANAGER=naive`
- `REWARD_MANAGER=naive_llm`
- `REWARD_MANAGER=execution_reward`
- `REWARD_MANAGER=info_reward_llm`

You can tune reward-related params by editing the `build_reward_args` section in:
- `examples/release/run_search_ppo_example.sh`
- `examples/release/run_search_grpo_example.sh`

## Retrieval service

If your tool configuration requires a local retrieval server, launch it first.

Example helper script:

```bash
CONDA_BIN_PATH=/path/to/conda_env/bin \
INDEX_PATH=/path/to/e5_Flat.index \
CORPUS_PATH=/path/to/wiki.jsonl \
bash examples/release/run_retriever_example.sh
```

Before training, verify health endpoint if your retriever serves one (commonly `http://localhost:8000/health`).

### Recommended defaults (aligned with verl docs)

The following defaults are consistent with the verl multi-turn search tool integration guide:

- Retriever model: `intfloat/e5-base-v2`
- Retriever name: `e5`
- Top-k: `3`
- Retrieval endpoint in tool config: `http://127.0.0.1:8000/retrieve`
- Health endpoint (for startup checks): `http://127.0.0.1:8000/health`

For tool config (`search_tool_config.yaml`), a practical default shape is:

```yaml
tools:
  - class_name: verl.tools.search_tool.SearchTool
    config:
      retrieval_service_url: http://127.0.0.1:8000/retrieve
      num_workers: 120
      rate_limit: 120
      timeout: 30
```

Reference: verl Search Tool Integration docs  
https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html#search-tool-integration

## Core codebase map

### Trainer and workflow

- Entry point: `verl/trainer/main_ppo.py`
- Main training loop: `verl/trainer/ppo/ray_trainer.py`
- Reward loading and dispatch: `verl/trainer/ppo/reward.py`
- Advantage and policy/value core algorithms: `verl/trainer/ppo/core_algos.py`
- Metrics and validation aggregation: `verl/trainer/ppo/metric_utils.py`

### Reward managers

- Base/simple manager: `verl/workers/reward_manager/naive_llm.py`
- Execution reward: `verl/workers/reward_manager/execution_reward_llm.py`
- Info reward variants:
  - `verl/workers/reward_manager/info_reward_llm.py`
  - `verl/workers/reward_manager/info_reward_llm_hmax.py`

### Worker execution

- FSDP worker stack: `verl/workers/fsdp_workers.py`
- Actor implementation details: `verl/workers/actor/dp_actor.py`

## Typical experiment workflow

1. Prepare dataset parquet files (`TRAIN_DATA`, `VAL_DATA`).
2. Prepare tool config (for multi-turn tool calls).
3. Start retrieval server (if required by tool config).
4. Pick algorithm:
   - PPO: `ADV_ESTIMATOR=gae`
   - GRPO/SGRPO: `ADV_ESTIMATOR=grpo` or `sgrpo`
5. Pick reward manager (`REWARD_MANAGER=...`).
6. Launch training with one of the release scripts.
7. Monitor:
   - console logs
   - W&B metrics (if configured)
   - saved rollout/checkpoint artifacts
8. Run validation/checkpoint evaluation with your internal scripts or a parameterized validation pipeline.

## Updating dependencies

After editing `pyproject.toml`:

```bash
uv sync
```

This keeps `.venv/` and `uv.lock` consistent.
