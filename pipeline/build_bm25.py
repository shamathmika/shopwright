import bm25s
import pandas as pd
import Stemmer

from shopwright.catalog_text import product_text
from shopwright.config import DATA

CATALOG = DATA / "catalog.parquet"
OUT = DATA / "bm25"

df = pd.read_parquet(CATALOG)
texts = df.apply(product_text, axis=1).tolist()

stemmer = Stemmer.Stemmer("english")
tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)

bm25 = bm25s.BM25(k1=1.2, b=0.75)
bm25.index(tokens)
bm25.save(str(OUT))

for q in ["GWTN156", "noise cancelling earbuds for running"]:
    qt = bm25s.tokenize([q], stopwords="en", stemmer=stemmer)
    ids, scores = bm25.retrieve(qt, k=3)
    print(f"\n{q!r}")
    for i, s in zip(ids[0], scores[0]):
        print(f"  {s:6.2f}  {df.iloc[i]['title'][:90]}")
