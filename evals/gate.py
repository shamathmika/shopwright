import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RUNS = EVAL_DIR / "runs"
CONFIG = EVAL_DIR / "gate.json"


def load(run: str) -> dict:
    return json.loads((RUNS / run / "summary.json").read_text())


def check(base: dict, cand: dict, cfg: dict) -> list[str]:
    failures = []
    if base.get("eval_set_sha") != cand.get("eval_set_sha"):
        failures.append("baseline and candidate used different eval sets")
    b, c = base["overall"], cand["overall"]
    if c["success"] < cfg["min_success"]:
        failures.append(f"success {c['success']:.3f} below minimum {cfg['min_success']}")
    if b["success"] - c["success"] > cfg["max_success_drop"]:
        failures.append(f"success dropped {b['success']:.3f} -> {c['success']:.3f}")
    bt, ct = base["by_split"]["test"]["success"], cand["by_split"]["test"]["success"]
    if bt - ct > cfg["max_test_success_drop"]:
        failures.append(f"test-split success dropped {bt:.3f} -> {ct:.3f}")
    if c["invented"] - b["invented"] > cfg["max_invented_increase"]:
        failures.append(f"invented asins rose {b['invented']:.3f} -> {c['invented']:.3f}")
    return failures


def main():
    cfg = json.loads(CONFIG.read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=cfg["baseline"])
    ap.add_argument("--candidate", default=cfg["candidate"])
    args = ap.parse_args()
    base, cand = load(args.baseline), load(args.candidate)
    failures = check(base, cand, cfg)
    print(f"gate: {args.baseline} (success {base['overall']['success']:.3f}) -> "
          f"{args.candidate} (success {cand['overall']['success']:.3f})")
    for f in failures:
        print(f"FAIL: {f}")
    if failures:
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
