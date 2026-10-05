import time
import faiss
import numpy as np
from shopwright.config import DATA

EMB = DATA / "embeddings.npy"
M = 32
EF_CONSTRUCTION = 200

emb = np.load(EMB)
d = emb.shape[1]

flat = faiss.IndexFlatIP(d)
flat.add(emb)
faiss.write_index(flat, str(DATA / "flat.index"))

hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = EF_CONSTRUCTION
t0 = time.time()
hnsw.add(emb)
print(f"HNSW build: {time.time() - t0:.1f}s")
faiss.write_index(hnsw, str(DATA / "hnsw.index"))
print(flat.ntotal, hnsw.ntotal)
