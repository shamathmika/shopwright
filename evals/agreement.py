import argparse
import json
from pathlib import Path

import pandas as pd

RUNS = Path(__file__).resolve().parent / "runs"
PASS = 4
DIMENSIONS = ["faithfulness", "overall"]


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    observed = (a == b).mean()
    expected = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float("nan") if expected == 1 else float((observed - expected) / (1 - expected))


def compare(human: pd.Series, judge: pd.Series) -> dict:
    h_pass, j_pass = human >= PASS, judge >= PASS
    return {
        "exact_agreement": round(float((human == judge).mean()), 3),
        "within_1": round(float(((human - judge).abs() <= 1).mean()), 3),
        "spearman": round(float(human.corr(judge, method="spearman")), 3),
        "pass_agreement": round(float((h_pass == j_pass).mean()), 3),
        "pass_kappa": round(cohen_kappa(h_pass, j_pass), 3),
        "mean_human": round(float(human.mean()), 2),
        "mean_judge": round(float(judge.mean()), 2),
        "judge_stricter_than_human": int((judge < human).sum()),
        "judge_looser_than_human": int((judge > human).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    run_dir = RUNS / args.run
    human = pd.DataFrame([json.loads(line) for line in open(run_dir / "human_labels.jsonl")])
    judged = [json.loads(line) for line in open(run_dir / "judge.jsonl")]
    judge = pd.DataFrame([{"id": j["id"], **{d: j["judge"][d] for d in DIMENSIONS}} for j in judged if j["judge"]])
    merged = human.merge(judge, on="id", suffixes=("_human", "_judge"))

    result = {"run": args.run, "n": len(merged), "pass_threshold": PASS,
              **{d: compare(merged[f"{d}_human"], merged[f"{d}_judge"]) for d in DIMENSIONS}}
    if len(merged) < 50:
        result["warning"] = f"only {len(merged)} labeled items; agreement numbers are noisy"
    disagreements = merged[(merged["overall_human"] - merged["overall_judge"]).abs() >= 2]
    result["large_disagreements"] = disagreements[["id", "overall_human", "overall_judge", "note"]].to_dict(orient="records")
    (run_dir / "agreement.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
