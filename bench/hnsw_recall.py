import time
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from shopwright.config import DATA, EMBED_MODEL, QUERY_PREFIX

N_QUERIES = 1000
K = 10
EF_VALUES = [8, 16, 32, 64, 128, 256]

titles = pd.read_parquet(DATA / "reviews.parquet", columns=["title"])["title"]
titles = titles[titles.str.len() > 15].sample(N_QUERIES, random_state=0).tolist()

model = SentenceTransformer(EMBED_MODEL, device="mps")
q = model.encode([QUERY_PREFIX + t for t in titles], normalize_embeddings=True).astype(np.float32)

faiss.omp_set_num_threads(1)
flat = faiss.read_index(str(DATA / "flat.index"))
hnsw = faiss.read_index(str(DATA / "hnsw.index"))

def run(index):
    index.search(q[:1], K)                      # warm-up
    ids, ms = np.empty((len(q), K), dtype=np.int64), []
    for i in range(len(q)):
        t0 = time.perf_counter()
        _, I = index.search(q[i:i + 1], K)
        ms.append((time.perf_counter() - t0) * 1000)
        ids[i] = I[0]
    return ids, np.array(ms)

truth, ms = run(flat)
print(f"{'flat':>8}  recall@10=1.000  p50={np.percentile(ms, 50):.3f}ms  p95={np.percentile(ms, 95):.3f}ms")
for ef in EF_VALUES:
    hnsw.hnsw.efSearch = ef
    ids, ms = run(hnsw)
    recall = np.mean([len(set(a) & set(b)) / K for a, b in zip(ids, truth)])
    print(f"ef={ef:>5}  recall@10={recall:.3f}  p50={np.percentile(ms, 50):.3f}ms  p95={np.percentile(ms, 95):.3f}ms")
