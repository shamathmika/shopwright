import argparse
import json
from pathlib import Path

import pandas as pd

from render import render_transcript

RUNS = Path(__file__).resolve().parent / "runs"
SAMPLE_SIZE = 50
SEED = 11
GUIDE = """Score 1-5, using the same rubric as the judge:
  faithfulness: 5 = every claim is in the tool results ... 1 = invented products/asins or contradicted claims
  overall:      would a careful shopper be well served? An unfaithful answer cannot score above 2.
Press Enter with no input to skip an item, or type q to quit (progress is saved)."""


def sample(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    frac = SAMPLE_SIZE / len(df)
    picked = df.groupby("category").sample(frac=min(1.0, frac), random_state=SEED)
    if len(picked) > SAMPLE_SIZE:
        picked = picked.sample(SAMPLE_SIZE, random_state=SEED)
    return picked.sort_values("id").to_dict(orient="records")


def ask(prompt: str) -> int | str | None:
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("", "q"):
            return raw or None
        if raw in {"1", "2", "3", "4", "5"}:
            return int(raw)
        print("  enter 1-5, Enter to skip, q to quit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    run_dir = RUNS / args.run
    rows = [json.loads(line) for line in open(run_dir / "results.jsonl")]
    out_path = run_dir / "human_labels.jsonl"
    done = {json.loads(line)["id"] for line in open(out_path)} if out_path.exists() else set()
    items = [r for r in sample(rows) if r["id"] not in done]
    print(f"{len(done)} labeled, {len(items)} to go\n{GUIDE}")

    with open(out_path, "a") as f:
        for n, row in enumerate(items, len(done) + 1):
            print("\n" + "=" * 100 + f"\n[{n}/{SAMPLE_SIZE}] {row['id']} ({row['category']})\n" + "=" * 100)
            print(render_transcript(row))
            faith = ask("\nfaithfulness 1-5: ")
            if faith == "q":
                break
            if faith is None:
                continue
            overall = ask("overall 1-5: ")
            if overall == "q":
                break
            if overall is None:
                continue
            note = input("note (optional): ").strip()
            f.write(json.dumps({"id": row["id"], "faithfulness": faith, "overall": overall, "note": note}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
