import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "evals" / "data" / "eval_set.jsonl"
OUT = ROOT / "bench" / "results"
MAX_TOKENS = 128


def prompts(n: int) -> list[str]:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    questions = [json.loads(line)["question"] for line in open(EVAL_SET)][:n]
    return [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False) for q in questions]


def sender(target: str, url: str, api_key: str):
    client = httpx.Client(base_url=url, timeout=120, headers={"Authorization": f"Bearer {api_key}"})
    if target == "vllm":
        def send(p):
            r = client.post("/v1/completions", json={"model": "qwen3-8b", "prompt": p, "max_tokens": MAX_TOKENS, "temperature": 0})
            return r.raise_for_status().json()["usage"]["completion_tokens"]
    else:
        def send(p):
            r = client.post("/v2/models/qwen3/generate",
                            json={"text_input": p, "parameters": {"max_tokens": MAX_TOKENS, "temperature": 0, "stream": False}})
            r.raise_for_status()
            return None
    return send


def run(send, items: list[str], workers: int) -> dict:
    latencies = []

    def timed(p):
        t0 = time.perf_counter()
        send(p)
        latencies.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(timed, items))
    wall = time.perf_counter() - t0
    return {"workers": workers, "requests": len(items), "p50_s": round(float(np.percentile(latencies, 50)), 3),
            "p95_s": round(float(np.percentile(latencies, 95)), 3), "throughput_rps": round(len(items) / wall, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["vllm", "triton"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    items = prompts(args.n)
    send = sender(args.target, args.url, args.api_key)
    send(items[0])
    result = {"target": args.target, "max_tokens": MAX_TOKENS,
              "levels": [run(send, items[:16], 1), run(send, items, 16)]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"llm_{args.target}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
