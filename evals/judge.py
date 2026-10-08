import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from render import render_transcript

EVAL_DIR = Path(__file__).resolve().parent
RUNS = EVAL_DIR / "runs"
load_dotenv(EVAL_DIR.parent / ".env")
JUDGE_MODEL = "claude-opus-5-5"
SCORES = ["faithfulness", "constraint_adherence", "helpfulness", "overall"]

RUBRIC = """You grade transcripts from a shopping assistant for an electronics store that sells only headphones, speakers, cameras and computers. You see the user's question, every tool call the assistant made with its result, and the assistant's final answer. Judge only against what the tools returned and what the user asked. Do not use outside knowledge about products.

Score each dimension from 1 to 5.

faithfulness: are the claims in the final answer supported by the tool results?
5 = every product, price, rating, spec and review claim appears in the tool results
4 = one minor unsupported detail that would not mislead a shopper
3 = some unsupported or slightly wrong details
2 = a materially wrong price, spec or review claim, or a comparison the tools do not support (for example "fewest complaints" after checking only one product)
1 = invented products or asins, or claims contradicted by the tool results

constraint_adherence: does the answer respect what the user asked for?
5 = every recommended product meets all stated constraints (type, brand, budget, rating), or the assistant clearly says nothing matches when that is true, or correctly declines an item the store does not sell
3 = mostly respects constraints but includes an item that violates one, or adds constraints the user never gave that change the result
1 = ignores the constraints, or presents products as matching when they do not

helpfulness: does it directly and usefully answer the question?
5 = answers exactly what was asked, concise, actionable
3 = partially answers, or buries the answer
1 = does not answer, or the answer is unusable

overall: would a careful shopper be well served by this answer? Weigh faithfulness most heavily: an unfaithful answer cannot score above 2 overall.

Write the rationale first, briefly citing the specific evidence, then give the scores."""

SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        **{s: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for s in SCORES},
    },
    "required": ["rationale", *SCORES],
    "additionalProperties": False,
}


def judge_one(client: anthropic.Anthropic, row: dict) -> dict:
    try:
        resp = client.beta.messages.create(
            model=JUDGE_MODEL,
            max_tokens=4000,
            system=RUBRIC,
            messages=[{"role": "user", "content": render_transcript(row)}],
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.APIStatusError as e:
        return {"id": row["id"], "judge": None, "error": f"{e.status_code}: {e.message}"}
    except anthropic.APIConnectionError as e:
        return {"id": row["id"], "judge": None, "error": f"connection: {e}"}
    if resp.stop_reason == "refusal":
        return {"id": row["id"], "judge": None, "error": "refusal"}
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return {"id": row["id"], "judge": json.loads(text), "judge_model": resp.model,
            "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}, "error": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    run_dir = RUNS / args.run
    rows = [json.loads(line) for line in open(run_dir / "results.jsonl")][: args.limit]
    out_path = run_dir / "judge.jsonl"
    done = {r["id"] for r in map(json.loads, open(out_path)) if r["judge"]} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"judging {len(todo)} of {len(rows)} with {JUDGE_MODEL}")

    client = anthropic.Anthropic(max_retries=5)
    tokens_in = tokens_out = 0
    with ThreadPoolExecutor(args.workers) as pool, open(out_path, "a") as f:
        for i, res in enumerate(pool.map(lambda r: judge_one(client, r), todo), 1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            if res["error"]:
                print(f"[{i}/{len(todo)}] {res['id']} ERROR {res['error']}")
                continue
            tokens_in += res["usage"]["input"]
            tokens_out += res["usage"]["output"]
            print(f"[{i}/{len(todo)}] {res['id']} " + " ".join(f"{s}={res['judge'][s]}" for s in SCORES))
    print(f"tokens: {tokens_in:,} in, {tokens_out:,} out")


if __name__ == "__main__":
    main()
