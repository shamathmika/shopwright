import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from shopwright.config import DATA, EMBED_MODEL, QUERY_PREFIX
from shopwright.embedding import TritonEmbedder, pick_device

OUT = Path(__file__).resolve().parent.parent / "bench" / "results" / "triton_vs_inprocess.json"


def latency(fn, queries) -> dict:
    fn(queries[:5])
    ms = []
    for q in queries:
        t0 = time.perf_counter()
        fn([q])
        ms.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": round(float(np.percentile(ms, 50)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2)}


def throughput(fn, queries, workers: int) -> float:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda q: fn([q]), queries))
    return round(len(queries) / (time.perf_counter() - t0), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triton-url", default="http://localhost:8001")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    titles = pd.read_parquet(DATA / "reviews.parquet", columns=["title"])["title"]
    queries = [QUERY_PREFIX + t for t in titles[titles.str.len() > 15].sample(args.n, random_state=0)]
    random.Random(0).shuffle(queries)

    st = SentenceTransformer(EMBED_MODEL, device=pick_device())
    triton = TritonEmbedder(args.triton_url)
    st_fn = lambda qs: st.encode(qs, normalize_embeddings=True)
    tr_fn = lambda qs: triton.encode(qs)

    a, b = st_fn(queries[:200]), tr_fn(queries[:200])
    result = {
        "device": pick_device(),
        "queries": args.n,
        "parity_min_cosine": round(float((a * b).sum(1).min()), 6),
        "in_process": {**latency(st_fn, queries), f"throughput_{args.workers}_threads_qps": throughput(st_fn, queries, args.workers)},
        "triton_http": {**latency(tr_fn, queries), f"throughput_{args.workers}_threads_qps": throughput(tr_fn, queries, args.workers)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
