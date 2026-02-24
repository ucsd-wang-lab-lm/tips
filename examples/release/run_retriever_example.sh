#!/usr/bin/env bash
set -euo pipefail

# Privacy-safe local retriever launcher.
# - No hardcoded private paths
# - No embedded credentials
# - Parameters via environment variables
#
# Usage:
#   chmod +x examples/release/run_retriever_example.sh
#   CONDA_BIN_PATH=/path/to/conda_env/bin \
#   INDEX_PATH=/path/to/e5_Flat.index \
#   CORPUS_PATH=/path/to/wiki.jsonl \
#   bash examples/release/run_retriever_example.sh
#
# Optional:
#   RETRIEVER_NAME=e5
#   RETRIEVER_MODEL=intfloat/e5-base-v2
#   TOPK=3
#   ENABLE_FAISS_GPU=true
#   HEALTH_CHECK=true
#   HEALTH_URL=http://localhost:8000/health

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_BIN_PATH="${CONDA_BIN_PATH:-}"
if [[ -z "${CONDA_BIN_PATH}" ]]; then
  echo "Please set CONDA_BIN_PATH, e.g. /opt/conda/envs/verl/bin"
  exit 1
fi
PYTHON_BIN="${CONDA_BIN_PATH%/}/python"

INDEX_PATH="${INDEX_PATH:-}"
CORPUS_PATH="${CORPUS_PATH:-}"
if [[ -z "${INDEX_PATH}" || -z "${CORPUS_PATH}" ]]; then
  echo "Please set INDEX_PATH and CORPUS_PATH."
  exit 1
fi

RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
TOPK="${TOPK:-3}"
ENABLE_FAISS_GPU="${ENABLE_FAISS_GPU:-true}"

HEALTH_CHECK="${HEALTH_CHECK:-true}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
MAX_RETRIES="${MAX_RETRIES:-30}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"

cd "${PROJECT_DIR}"

CMD=(
  "${PYTHON_BIN}"
  "examples/sglang_multiturn/search_r1_like/local_dense_retriever/old_rs.py"
  --index_path "${INDEX_PATH}"
  --corpus_path "${CORPUS_PATH}"
  --topk "${TOPK}"
  --retriever_name "${RETRIEVER_NAME}"
  --retriever_model "${RETRIEVER_MODEL}"
)

if [[ "${ENABLE_FAISS_GPU}" == "true" ]]; then
  CMD+=(--faiss_gpu)
fi

echo "Launching retriever..."
echo "  index_path: ${INDEX_PATH}"
echo "  corpus_path: ${CORPUS_PATH}"
echo "  retriever_name: ${RETRIEVER_NAME}"
echo "  retriever_model: ${RETRIEVER_MODEL}"
echo "  topk: ${TOPK}"
echo "  faiss_gpu: ${ENABLE_FAISS_GPU}"

"${CMD[@]}" > retriever_launch.log 2>&1 &
RETRIEVER_PID=$!
echo "Retriever started in background. PID=${RETRIEVER_PID}, log=./retriever_launch.log"

if [[ "${HEALTH_CHECK}" == "true" ]]; then
  echo "Checking retriever health: ${HEALTH_URL}"
  retry_count=0
  while true; do
    if curl -s "${HEALTH_URL}" | grep -q '"status":"ok"'; then
      echo "Retriever is healthy."
      break
    fi
    retry_count=$((retry_count + 1))
    if [[ "${retry_count}" -ge "${MAX_RETRIES}" ]]; then
      echo "Retriever failed health check after ${MAX_RETRIES} retries."
      exit 1
    fi
    sleep "${SLEEP_SECONDS}"
  done
fi

wait "${RETRIEVER_PID}"

