import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from checks import grade

EVAL_DIR = Path(__file__).resolve().parent
RUNS = EVAL_DIR / "runs"
EVAL_SET = EVAL_DIR / "data" / "eval_set.jsonl"
WEIGHTS = {"success": 0.4, "tool_correct": 0.2, "judge_overall": 0.3, "efficiency": 0.1}
JUDGE_SCORES = ["faithfulness", "constraint_adherence", "helpfulness", "overall"]


def load_rows(run_dir: Path) -> pd.DataFrame:
    results = [json.loads(line) for line in open(run_dir / "results.jsonl")]
    judged = {}
    if (run_dir / "judge.jsonl").exists():
        for line in open(run_dir / "judge.jsonl"):
            j = json.loads(line)
            if j["judge"]:
                judged[j["id"]] = j["judge"]
    records = []
    for r in results:
        result = {"answer": r["answer"], "trace": r["trace"], "steps": r.get("steps", r["grade"]["steps"]),
                  "hit_limit": r.get("hit_limit", r["grade"]["hit_limit"])}
        g = grade(r, result) if r["run_error"] is None else r["grade"]
        rec = {"id": r["id"], "category": r["category"], "latency_ms": r["latency_ms"],
               "run_error": r["run_error"] is not None, **g, "invented": bool(g["invented_asins"])}
        j = judged.get(r["id"])
        for s in JUDGE_SCORES:
            rec[f"judge_{s}"] = j[s] if j else None
        records.append(rec)
    df = pd.DataFrame(records)
    splits = {json.loads(line)["id"]: json.loads(line).get("split", "all") for line in open(EVAL_SET)}
    df["split"] = df["id"].map(splits).fillna("all")
    df["judge_overall_norm"] = (df["judge_overall"] - 1) / 4
    df["composite"] = (WEIGHTS["success"] * df["success"] + WEIGHTS["tool_correct"] * df["tool_correct"]
                       + WEIGHTS["judge_overall"] * df["judge_overall_norm"] + WEIGHTS["efficiency"] * df["efficiency"])
    no_judge = {k: v for k, v in WEIGHTS.items() if k != "judge_overall"}
    df["composite_no_judge"] = sum(w * df[k] for k, w in no_judge.items()) / sum(no_judge.values())
    return df


def build_summary(run_dir: Path) -> dict:
    df = load_rows(run_dir)
    metrics = ["success", "tool_correct", "steps", "efficiency", "tool_errors", "guessed_ids", "hit_limit", "invented",
               "run_error", "composite_no_judge"]
    judged = df["judge_overall"].notna()
    judge_cols = [f"judge_{s}" for s in JUDGE_SCORES]
    if judged.any():
        metrics += judge_cols + ["composite"]
    by_cat = df.groupby("category")[metrics + ["latency_ms"]].mean().round(3)
    by_cat["n"] = df.groupby("category").size()
    old = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    summary = {
        **{k: old[k] for k in ("run", "created", "commit", "model", "llm_host", "workers", "eval_set_sha") if k in old},
        "scored": datetime.now(timezone.utc).isoformat(),
        "n": len(df),
        "n_judged": int(judged.sum()),
        "weights": WEIGHTS,
        "overall": {m: round(float(df[m].mean()), 3) for m in metrics},
        "latency_ms": {"p50": float(df["latency_ms"].quantile(0.5)), "p95": float(df["latency_ms"].quantile(0.95))},
        "by_category": by_cat.reset_index().to_dict(orient="records"),
        "by_split": df.groupby("split")[metrics].mean().round(3).to_dict(orient="index"),
    }
    if judged.any() and not judged.all():
        summary["warning"] = f"composite uses only the {int(judged.sum())} judged rows"
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def print_summary(summary: dict) -> None:
    print(json.dumps({k: summary[k] for k in ("n", "n_judged", "overall", "latency_ms")}, indent=2))
    cols = ["category", "n", "success", "tool_correct", "steps", "invented", "composite_no_judge"]
    if summary["n_judged"]:
        cols += ["judge_faithfulness", "judge_overall", "composite"]
    print(pd.DataFrame(summary["by_category"])[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare")
    args = ap.parse_args()
    summary = build_summary(RUNS / args.run)
    print_summary(summary)
    if args.compare:
        base = build_summary(RUNS / args.compare)
        if base.get("eval_set_sha") != summary.get("eval_set_sha"):
            print("WARNING: runs used different eval sets; the comparison is not valid")
        keys = [k for k in summary["overall"] if k in base["overall"]]
        diff = pd.DataFrame({"base": [base["overall"][k] for k in keys],
                             "new": [summary["overall"][k] for k in keys]}, index=keys)
        diff["delta"] = (diff["new"] - diff["base"]).round(3)
        print(f"\n{args.compare} -> {args.run}\n{diff.to_string()}")
        old, new = load_rows(RUNS / args.compare).set_index("id"), load_rows(RUNS / args.run).set_index("id")
        fixed = int((~old["success"] & new["success"]).sum())
        broke = int((old["success"] & ~new["success"]).sum())
        print(f"per-question: {fixed} fixed, {broke} regressed, {len(old) - fixed - broke} unchanged")


if __name__ == "__main__":
    main()
