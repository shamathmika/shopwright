import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from shopwright.catalog_text import product_text
from shopwright.config import DATA, EMBED_MODEL

CATALOG = DATA / "catalog.parquet"
OUT = DATA / "embeddings.npy"

df = pd.read_parquet(CATALOG)
texts = df.apply(product_text, axis=1).tolist()

model = SentenceTransformer(EMBED_MODEL, device="mps")
emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                   show_progress_bar=True, convert_to_numpy=True)
emb = emb.astype(np.float32)
np.save(OUT, emb)
print(emb.shape, np.linalg.norm(emb[:3], axis=1))
