#!/bin/bash
set -euo pipefail

APP=/workspace/shopwright
VLLM=/workspace/venv/bin/vllm
PY=/workspace/app-venv/bin/python
export HF_HOME=/workspace/hf
export VLLM_API_KEY=${VLLM_API_KEY:-$(openssl rand -hex 16)}
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_API_KEY=$VLLM_API_KEY
export LLM_MODEL=qwen3-8b
export LLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
COMMON="--served-model-name qwen3-8b --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes --port 8000"
cd $APP
mkdir -p bench/results

start_vllm() {
  local model=$1 extra=$2 tag=$3
  pkill -f "vllm serve" || true
  while curl -sf localhost:8000/health >/dev/null 2>&1; do sleep 2; done
  sleep 5
  echo "=== starting vLLM: $tag"
  nohup $VLLM serve "$model" $COMMON $extra > bench/results/vllm_$tag.log 2>&1 &
  until curl -sf localhost:8000/health >/dev/null 2>&1; do sleep 5; done
  grep -E "GPU KV cache size|Available KV cache memory|weights" bench/results/vllm_$tag.log | tail -3 || true
}

start_api() {
  pkill -f "uvicorn shopwright.api" || true
  nohup $PY -m uvicorn shopwright.api:app --host 0.0.0.0 --port 8010 > bench/results/api.log 2>&1 &
  until curl -sf localhost:8010/health >/dev/null 2>&1; do sleep 3; done
  echo "=== API up"
}

start_vllm Qwen/Qwen3-8B "" bf16-cache-on
start_api
$PY bench/load_test.py --label bf16-prefix-cache-on
$PY evals/run_eval.py --name vllm-bf16 --workers 8 | tail -12

start_vllm Qwen/Qwen3-8B "--no-enable-prefix-caching" bf16-cache-off
$PY bench/load_test.py --label bf16-prefix-cache-off

start_vllm Qwen/Qwen3-8B-AWQ "" awq-cache-on
$PY bench/load_test.py --label awq-prefix-cache-on
$PY evals/run_eval.py --name vllm-awq --workers 8 | tail -12

tar czf /workspace/phase4_results.tar.gz bench/results evals/runs/vllm-bf16 evals/runs/vllm-awq
echo "=== DONE: download /workspace/phase4_results.tar.gz"
