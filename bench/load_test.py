import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "evals" / "data" / "eval_set.jsonl"
OUT = ROOT / "bench" / "results"
LEVELS = [1, 8, 32]
REQUESTS_PER_USER = 3
MIN_REQUESTS = 20
SEED = 3


async def one_request(client: httpx.AsyncClient, url: str, question: str) -> dict:
    t0 = time.perf_counter()
    ttft, tool_errors, done, error = None, 0, None, None
    try:
        async with client.stream("POST", url, json={"question": question}) as r:
            if r.status_code != 200:
                error = f"http {r.status_code}"
            else:
                event = None
                async for line in r.aiter_lines():
                    if line.startswith("event: "):
                        event = line[7:]
                    elif line.startswith("data: "):
                        data = json.loads(line[6:])
                        if event == "token" and ttft is None:
                            ttft = time.perf_counter() - t0
                        elif event == "tool_result" and data["error"]:
                            tool_errors += 1
                        elif event == "done":
                            done = data
                        elif event == "error":
                            error = data["message"]
    except httpx.HTTPError as e:
        error = f"{type(e).__name__}: {e}"
    return {"latency_s": time.perf_counter() - t0, "ttft_s": ttft, "tool_errors": tool_errors,
            "ok": done is not None and error is None, "error": error,
            "steps": done["steps"] if done else None}


async def user(client: httpx.AsyncClient, url: str, queue: asyncio.Queue, results: list) -> None:
    while not queue.empty():
        question = queue.get_nowait()
        results.append(await one_request(client, url, question))


def pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 2) if values else None


def summarize(users: int, results: list[dict], wall_s: float) -> dict:
    ok = [r for r in results if r["ok"]]
    latencies = [r["latency_s"] for r in ok]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    with_tool_errors = [r for r in results if r["tool_errors"]]
    return {
        "users": users,
        "requests": len(results),
        "error_rate": round(1 - len(ok) / len(results), 3),
        "latency_p50_s": pct(latencies, 50),
        "latency_p95_s": pct(latencies, 95),
        "ttft_p50_s": pct(ttfts, 50),
        "ttft_p95_s": pct(ttfts, 95),
        "throughput_rps": round(len(ok) / wall_s, 3),
        "mean_steps": round(float(np.mean([r["steps"] for r in ok])), 2) if ok else None,
        "requests_with_tool_errors": len(with_tool_errors),
        "recovered_after_tool_error": sum(r["ok"] for r in with_tool_errors),
        "errors": sorted({r["error"] for r in results if r["error"]})[:5],
    }


async def run_level(url: str, users: int, questions: list[str]) -> dict:
    n = max(MIN_REQUESTS, users * REQUESTS_PER_USER)
    queue: asyncio.Queue = asyncio.Queue()
    for q in questions[:n]:
        queue.put_nowait(q)
    results: list[dict] = []
    t0 = time.perf_counter()
    limits = httpx.Limits(max_connections=users + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), limits=limits) as client:
        await asyncio.gather(*(user(client, url, queue, results) for _ in range(users)))
    return summarize(users, results, time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", default="http://localhost:8010/chat/stream")
    ap.add_argument("--levels", type=int, nargs="*", default=LEVELS)
    args = ap.parse_args()

    questions = [json.loads(line)["question"] for line in open(EVAL_SET)]
    random.Random(SEED).shuffle(questions)
    rows = []
    for users in args.levels:
        row = asyncio.run(run_level(args.url, users, questions))
        rows.append(row)
        print(json.dumps(row), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.label}.json").write_text(json.dumps({"label": args.label, "levels": rows}, indent=2))


if __name__ == "__main__":
    main()
