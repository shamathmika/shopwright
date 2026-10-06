from dataclasses import dataclass

import bm25s
import faiss
import numpy as np
import pandas as pd
import Stemmer

from shopwright.config import DATA, QUERY_PREFIX
from shopwright.embedding import load_embedder

RRF_K = 60


@dataclass
class Hit:
    row: int
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


class Searcher:
    def __init__(self, index: str = "flat", ef_search: int = 64):
        self.catalog = pd.read_parquet(DATA / "catalog.parquet")
        self.emb = np.load(DATA / "embeddings.npy")
        self.model = load_embedder()
        self.index = faiss.read_index(str(DATA / f"{index}.index"))
        if index == "hnsw":
            self.index.hnsw.efSearch = ef_search
        self.bm25 = bm25s.BM25.load(str(DATA / "bm25"))
        self.stemmer = Stemmer.Stemmer("english")

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]

    def dense(self, query: str, n: int) -> list[int]:
        _, ids = self.index.search(self.embed_query(query)[None, :], n)
        return [int(i) for i in ids[0] if i != -1]

    def keyword(self, query: str, n: int) -> list[int]:
        qt = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, show_progress=False)
        ids, scores = self.bm25.retrieve(qt, k=n, show_progress=False)
        return [int(i) for i, s in zip(ids[0], scores[0]) if s > 0]

    def search(self, query: str, k: int = 10, n_candidates: int = 100) -> list[Hit]:
        hits: dict[int, Hit] = {}
        for rank, row in enumerate(self.dense(query, n_candidates), 1):
            hits[row] = Hit(row, 1 / (RRF_K + rank), dense_rank=rank)
        for rank, row in enumerate(self.keyword(query, n_candidates), 1):
            h = hits.setdefault(row, Hit(row, 0.0))
            h.score += 1 / (RRF_K + rank)
            h.bm25_rank = rank
        return sorted(hits.values(), key=lambda h: h.score, reverse=True)[:k]
