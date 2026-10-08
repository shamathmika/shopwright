# ShopWright

A shopping assistant agent over a real Amazon product catalog, served on a self-hosted LLM, with an offline evaluation harness that gates changes.

- **Agent:** ReAct loop in LangGraph with 4 tools (catalog search, attribute filter, product compare, review summary), behind an async FastAPI service that streams tool steps and tokens over SSE.
- **Model:** Qwen3-8B served with vLLM on one rented NVIDIA A40 (48 GB), full precision (BF16) and AWQ 4-bit.
- **Eval:** 200 questions with ground truth derived from the catalog, code-graded metrics, and an LLM judge (Claude Opus 5.5) checked against 50 human labels.

Every number below comes from a run in this repo (`evals/runs/`, `bench/results/`). Sample sizes are small and everything ran on a single GPU; see [Limitations](#limitations).

## Data

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD), Electronics category, filtered to four product types (headphones, speakers, cameras, computers), products with a price and at least 5 ratings, exact-duplicate titles removed: **29,819 products** and **421,619 reviews** (up to 30 most-helpful per product).

To rebuild: download `meta_Electronics.jsonl` and `Electronics.jsonl` into `data/`, then run the scripts in `pipeline/` in order: `build_catalog.py`, `build_reviews.py`, `embed.py`, `build_index.py`, `build_bm25.py`.

## Architecture

```
client ──SSE──► FastAPI (/chat/stream) ──► LangGraph agent ──► vLLM (Qwen3-8B, OpenAI-compatible API)
                                              │
                                              └──► tools: search · filter · compare · review_summary
                                                     │
                                                     ├── hybrid search: bge-small embeddings (exact flat index) + BM25, fused with RRF
                                                     └── review_summary makes its own LLM call over up to 15 reviews
```

| Path | What |
|---|---|
| `pipeline/` | offline data build: catalog, reviews, embeddings, indexes |
| `shopwright/` | runtime package: search, tools, agent, API |
| `evals/` | eval set, graders, judge, human labeling, agreement, CI gate (see `evals/README.md`) |
| `bench/` | HNSW recall benchmark, load test, vLLM benchmark matrix |
| `tests/` | unit tests run in CI |

## Results

### Retrieval: exact vs approximate nearest neighbours

29.8K vectors, 384 dims, M-series CPU, 1 thread, 1,000 queries.

| Index | recall@10 | p50 latency |
|---|---|---|
| Flat (exact) | 1.000 | 0.58 ms |
| HNSW, M=32, efSearch=64 | 0.731 on review-title queries, 0.997 on catalog-vector queries | 0.08 ms |

The low recall on review-title queries comes from near-ties: vague queries have dozens of almost equally similar products (mean similarity lost: 0.004). The agent uses exact search, because 0.6 ms is negligible next to multi-second LLM calls at this catalog size.

### Agent quality (200-question eval, A40)

| Model | Task success | Test split (100) | Tool-call correct | Mean steps | Judge overall (1–5) | Judge faithfulness (1–5) | Composite |
|---|---|---|---|---|---|---|---|
| Qwen3-8B BF16 | **0.72** | 0.74 | 0.76 | 2.4 | 3.50 | 4.07 | **0.723** |
| Qwen3-8B AWQ 4-bit | **0.52** | 0.53 | 0.55 | 2.0 | not judged | not judged | — |

Composite = 0.4 × success + 0.2 × tool correctness + 0.3 × judge overall (scaled 0–1) + 0.1 × step efficiency.

Per category (BF16): out-of-scope 1.00, lookup 0.93, constraint 0.82, reviews 0.77, multi-constraint 0.67, impossible 0.53, **compare 0.20**. The main failure is the model guessing product IDs (including a format placeholder from its prompt) instead of searching first: 0.41 guessed IDs per question.

### LLM judge vs human labels

Judge: Claude Opus 5.5 with a 4-dimension rubric. Human: one labeler (the author), blind to the judge's scores, on 50 answers sampled across categories.

| | Overall | Faithfulness |
|---|---|---|
| Cohen's κ (pass ≥ 4) | 0.63 | 0.52 |
| Spearman ρ | 0.71 | 0.45 |
| Within 1 point | 88% | 86% |
| Judge stricter / looser than human | 14 / 13 | 20 / 3 |

The judge's overall score tracks the human's in aggregate. On faithfulness it is systematically harsher. Judging all 200 answers used 341K input and 81K output tokens (about $3).

### Serving: load test (A40, API and load generator on the same machine)

Each level sends 20, 24 and 96 requests from 1, 8 and 32 concurrent users through `/chat/stream`. Each request is a full agent run (tool calls plus generation).

| Configuration | Users | p50 latency | p95 latency | p50 time to first token | Throughput | Error rate |
|---|---|---|---|---|---|---|
| BF16, prefix cache on | 1 | 6.2 s | 10.5 s | 1.2 s | 0.17 req/s | 0% |
| | 8 | 6.4 s | 12.4 s | 1.3 s | 0.91 req/s | 0% |
| | 32 | 16.4 s | 28.2 s | 3.5 s | 1.53 req/s | 0% |
| BF16, prefix cache off | 1 | 6.4 s | 10.9 s | 1.5 s | 0.17 req/s | 0% |
| | 8 | 9.3 s | 17.2 s | 2.7 s | 0.66 req/s | 0% |
| | 32 | 24.3 s | 45.0 s | 7.7 s | 1.06 req/s | 0% |
| AWQ 4-bit, prefix cache on | 1 | 1.7 s | 3.3 s | 0.5 s | 0.55 req/s | 0% |
| | 8 | 3.4 s | 10.4 s | 0.9 s | 1.55 req/s | 0% |
| | 32 | 6.5 s | 18.9 s | 3.0 s | 2.65 req/s | 0% |

- **Prefix caching** makes no difference with 1 user but matters under load: at 32 users, p50 drops from 24.3 s to 16.4 s and throughput rises 44%. Every agent request repeats the same system prompt and tool schemas, so the shared prefix is large.
- **AWQ** cut weight memory from 15.5 GB to 6.0 GB and raised KV-cache capacity from 157K to 219K tokens. Its latency advantage is partly real and partly because it takes fewer steps: it often gives up early instead of searching. It also lost 20 points of task success, and the CI gate rejects it.
- **Tool-failure retries:** every request that hit a tool error still completed (27 of 27 at 32 users with BF16).

## CI gate

`.github/workflows/ci.yml` runs unit tests (tool schemas, dispatcher error handling, graders) and `evals/gate.py`. The gate fails if a committed eval run drops more than 3 points of task success (overall or test split), falls below 0.65, invents more product IDs than the baseline, or used a different eval set. CI has no GPU, so it does not run the agent: an eval run happens on the GPU, its results are committed, and CI enforces the thresholds. Against the BF16 baseline, the AWQ run fails the gate on three checks.

## Limitations

- **One GPU, small samples.** One A40, 20–96 requests per load level, 200 eval questions, 50 human labels from a single labeler. These are measurements of this setup, not scaling results.
- **The load generator ran on the same machine as the API**, so latencies exclude network time.
- **Prefix caching was not re-evaluated for quality.** It should not change outputs, but that was not measured.
- **AWQ was not judged**, and its latency is not like-for-like with BF16, because it takes fewer steps.
- **Ground truth inherits seller data errors** (e.g. wrong specs in a listing).
- **Graders are heuristics**: product mentions are detected from IDs, title prefixes and model names, and "nothing matches" answers by a phrase regex.
- **Prompt tuning was done on the dev split** with Ollama on a laptop (a different 4-bit quantization). Full precision on vLLM scored 0.72 overall with the same prompt.
