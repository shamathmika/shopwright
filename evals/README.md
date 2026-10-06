# Offline evaluation

200 shopping questions with ground truth derived from the catalog, graded by code where possible and by an LLM judge where code can't decide.

## Pipeline

| Step | Command | Output |
|---|---|---|
| Build the eval set (deterministic, seeded) | `python evals/build_eval_set.py` | `evals/data/eval_set.jsonl` |
| Run the agent on every question | `python evals/run_eval.py --name <run>` | `evals/runs/<run>/results.jsonl`, `summary.json` |
| Judge the answers (needs `ANTHROPIC_API_KEY`) | `python evals/judge.py --run <run>` | `evals/runs/<run>/judge.jsonl` |
| Recompute all scores | `python evals/report.py --run <run> [--compare <base>]` | updated `summary.json` |
| Hand-label 50 sampled answers (blind to the judge) | `python evals/label.py --run <run>` | `evals/runs/<run>/human_labels.jsonl` |
| Judge vs human agreement | `python evals/agreement.py --run <run>` | `evals/runs/<run>/agreement.json` |

`run_eval.py` accepts `--limit-per-category N` for quick smoke runs and `--workers N` for concurrent requests (use 1 against Ollama, which serves one request at a time).

## Question categories

| Category | n | Ground truth | Task success means |
|---|---|---|---|
| constraint | 60 | kind, brand, price range, min rating, title pattern | at least one product recommended, and every recommended product meets every constraint |
| lookup | 30 | one target asin (named by a unique model code) | the target is recommended |
| compare | 30 | two target asins of the same kind | `compare_products` called with both, and both mentioned |
| reviews | 30 | target asin + aspect | `review_summary` called on the target, and the target mentioned |
| multi | 15 | constraints + aspect | constraint success, and `review_summary` called on a product that meets them |
| out_of_scope | 20 | the store does not sell it | nothing recommended, or the answer says it isn't sold |
| impossible | 15 | constraints with zero catalog matches | nothing recommended, or the answer says nothing matches |

Each item has a `split` (dev or test, about 50/50 per category). Tune prompts on dev failures; report test.

## Metrics

| Metric | Source |
|---|---|
| success | per-category rule above |
| tool_correct | the right tool called with the right key arguments (e.g. filter `kind`, `max_price` within 1%, `brand`) |
| steps, efficiency | agent turns; efficiency loses 0.25 per step beyond the category's expected count |
| tool_errors, invented | tool calls that returned an error; asins in the answer that no tool returned |
| judge_* | Claude Opus 5.5 scores 1-5 for faithfulness, constraint adherence, helpfulness, overall |
| composite | 0.4 success + 0.2 tool_correct + 0.3 judge overall (scaled 0-1) + 0.1 efficiency |
| composite_no_judge | the same without the judge term, reweighted to 0-1 |

## Known limitations

- Ground truth uses seller-entered catalog fields, which are sometimes wrong (e.g. a laptop listing claiming 256 GB when the title says 128 GB).
- "Recommended" is detected from asins in the answer, plus title-prefix and model-name matches. An answer that names a product loosely without its asin can be missed.
- The impossible and out_of_scope rules accept a regex match on phrases like "couldn't find", so a confused answer containing such a phrase can pass.
- Rerunning identical code on Ollama is nearly deterministic, but any prompt change flips many unrelated answers in both directions. Compare per-question fixes and regressions (`report.py --compare`), not only the averages. vLLM with batching may be less deterministic; measure its rerun variance before trusting small deltas there.
- The judge sees the same rendered transcript a human labeler sees, but it is not validated until `agreement.py` has been run on 50 hand labels.

## Results so far (local development runs)

Agent: Qwen3-8B via Ollama on an M5 Pro (Ollama's default 4-bit quantization), one request at a time. **These are development numbers, not serving numbers.** The reportable runs are on vLLM; see the top-level README. Judge scores are not included yet (no judge run).

| Run | Change | Success (all 200) | Success (test 100) | Tool correct | Guessed ids / question | p50 / p95 latency |
|---|---|---|---|---|---|---|
| v0 | baseline prompt | 0.720 | 0.752 | 0.775 | 0.68 | 5.6 s / 11.4 s |
| v1 | "search before using an asin", "use review_summary for review questions", no example asin, treat brand "None" as no brand | 0.720 | 0.703 | 0.800 | 0.62 | 6.0 s / 11.8 s |
| v2 | v1 + a concrete citation format example with a fake placeholder, broader review rule | 0.760 | 0.743 | 0.815 | 0.45 | 5.6 s / 12.2 s |
| v2 repeat | none (same code rerun) | 0.755 | 0.743 | 0.810 | 0.45 | 5.7 s / 12.5 s |

What these runs show:
- Rerunning identical code changed 1 of 200 outcomes, so on this setup a 0.5-point move is noise and larger moves come from the change itself.
- v2 is +4 points overall but -0.9 on the held-out test split. The gain is concentrated in reviews (0.60 to 0.83) and multi (0.33 to 0.67), while compare fell (0.30 to 0.20). It is not a clear improvement on unseen questions.
- The main remaining failure is compare: the model passes model names, or the format placeholder, as asins (2 guessed ids per compare question) instead of searching for each product first.
