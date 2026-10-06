import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from checks import grade
from report import build_summary, print_summary
from shopwright.agent import run_agent
from shopwright.config import LLM_BASE_URL, LLM_MODEL
from shopwright.resources import get_reviews, get_searcher

EVAL_DIR = Path(__file__).resolve().parent
EVAL_SET = EVAL_DIR / "data" / "eval_set.jsonl"
RUNS = EVAL_DIR / "runs"


def load_items(limit_per_category: int | None, categories: list[str] | None) -> list[dict]:
    items = [json.loads(line) for line in open(EVAL_SET)]
    if categories:
        items = [it for it in items if it["category"] in categories]
    if limit_per_category:
        seen: dict[str, int] = {}
        kept = []
        for it in items:
            if seen.get(it["category"], 0) < limit_per_category:
                kept.append(it)
                seen[it["category"]] = seen.get(it["category"], 0) + 1
        items = kept
    return items


def run_one(item: dict) -> dict:
    t0 = time.perf_counter()
    try:
        result = run_agent(item["question"])
        error = None
    except Exception as e:
        result = {"answer": "", "steps": 0, "trace": [], "hit_limit": False}
        error = f"{type(e).__name__}: {e}"
    latency = round((time.perf_counter() - t0) * 1000)
    graded = grade(item, result) if error is None else {
        "success": False, "tool_correct": False, "steps": 0, "efficiency": 0.0, "tool_errors": 0,
        "hit_limit": False, "recommended": [], "invented_asins": []}
    return {**item, "answer": result["answer"], "trace": result["trace"], "steps": result["steps"],
            "hit_limit": result["hit_limit"], "latency_ms": latency,
            "run_error": error, "grade": graded}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=EVAL_DIR, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit-per-category", type=int)
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    items = load_items(args.limit_per_category, args.categories)
    out_dir = RUNS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    get_searcher()
    get_reviews()

    rows = []
    with ThreadPoolExecutor(args.workers) as pool, open(out_dir / "results.jsonl", "w") as f:
        for i, row in enumerate(pool.map(run_one, items), 1):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            g = row["grade"]
            print(f"[{i}/{len(items)}] {row['id']} {row['category']:<12} success={g['success']!s:<5} "
                  f"tool={g['tool_correct']!s:<5} steps={g['steps']} {row['latency_ms']}ms", flush=True)

    meta = {"run": args.name, "created": datetime.now(timezone.utc).isoformat(), "commit": git_commit(),
            "model": LLM_MODEL, "llm_host": urlparse(LLM_BASE_URL).hostname, "workers": args.workers,
            "eval_set_sha": hashlib.sha256(EVAL_SET.read_bytes()).hexdigest()[:12]}
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    print_summary(build_summary(out_dir))


if __name__ == "__main__":
    main()
