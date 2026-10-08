#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

TRITON_IMAGE=${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.08-py3}
TRITON_VLLM_IMAGE=${TRITON_VLLM_IMAGE:?set to a recent nvcr.io/nvidia/tritonserver:<yy.mm>-vllm-python-py3 tag (Qwen3 needs vLLM 0.8.5+)}
BENCH=/tmp/bench-venv/bin/python
DEVICE_PLUGIN=https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

command -v nvidia-smi >/dev/null || { echo "Run this on the GPU machine (ssh ubuntu@<IP>), not on your Mac."; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv
command -v nvidia-container-runtime >/dev/null || { echo "install nvidia-container-toolkit first"; exit 1; }

if ! command -v k3s >/dev/null; then
  curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
fi
until kubectl get nodes 2>/dev/null | grep -q " Ready"; do sleep 3; done
kubectl apply -f k8s/gpu/runtimeclass.yaml
curl -sfL "$DEVICE_PLUGIN" | sed 's/^    spec:$/    spec:\n      runtimeClassName: nvidia/' | kubectl apply -f -
until kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' | grep -q 1; do sleep 3; done
echo "=== GPU visible to Kubernetes"

for svc in agent retrieval; do
  docker build -t shopwright-$svc:latest -f docker/$svc.Dockerfile .
  docker save shopwright-$svc:latest | sudo k3s ctr images import -
done
docker build -t shopwright-web:latest web
docker save shopwright-web:latest | sudo k3s ctr images import -

kubectl create namespace shopwright --dry-run=client -o yaml | kubectl apply -f -
kubectl -n shopwright get secret llm >/dev/null 2>&1 || \
  kubectl -n shopwright create secret generic llm --from-literal=LLM_API_KEY="${LLM_API_KEY:-$(openssl rand -hex 16)}"
kubectl apply -k k8s/gpu
kubectl -n shopwright rollout status deploy/retrieval deploy/agent deploy/web --timeout=15m
kubectl -n shopwright rollout status deploy/vllm --timeout=30m
kubectl -n shopwright get pods -o wide
echo "=== web UI: http://<this-machine>:30300 (or ssh -L 30300:localhost:30300 and open http://localhost:30300)"

if [ "${RUN_TRITON:-1}" = 1 ]; then
  docker rm -f triton >/dev/null 2>&1 || true
  docker run -d --name triton --gpus all -p 8001:8000 -v "$PWD/serving/models:/models" "$TRITON_IMAGE" \
    tritonserver --model-repository=/models
  until curl -sf localhost:8001/v2/health/ready >/dev/null; do sleep 3; done
  python3 -m venv /tmp/bench-venv
  /tmp/bench-venv/bin/pip install -q -r requirements.txt
  /tmp/bench-venv/bin/pip install -q -e . --no-deps
  LLM_BASE_URL=x LLM_API_KEY=x LLM_MODEL=x LLM_EXTRA_BODY='{}' $BENCH serving/bench_triton.py
  docker rm -f triton
fi

echo "=== LLM: vLLM in Kubernetes vs Triton vLLM backend"
KEY=$(kubectl -n shopwright exec deploy/vllm -- printenv VLLM_API_KEY)
kubectl -n shopwright port-forward svc/vllm 8000:8000 >/dev/null 2>&1 &
PF=$!
sleep 3
$BENCH serving/bench_llm.py --target vllm --url http://localhost:8000 --api-key "$KEY"
kill $PF 2>/dev/null || true
kubectl -n shopwright scale deploy/vllm --replicas=0
kubectl -n shopwright wait --for=delete pod -l app=vllm --timeout=5m || true
docker run -d --name triton-vllm --gpus all --shm-size=2g -p 8002:8000 -v "$PWD/serving/triton_vllm:/models" \
  -v /var/lib/shopwright/hf:/root/.cache/huggingface "$TRITON_VLLM_IMAGE" tritonserver --model-repository=/models
until curl -sf localhost:8002/v2/health/ready >/dev/null; do
  docker ps -q -f name=triton-vllm | grep -q . || { echo "triton-vllm exited"; docker logs --tail 30 triton-vllm; exit 1; }
  sleep 5
done
$BENCH serving/bench_llm.py --target triton --url http://localhost:8002
docker rm -f triton-vllm
kubectl -n shopwright scale deploy/vllm --replicas=1
echo "=== DONE. Results in bench/results/"
