import os

import httpx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from shopwright.config import EMBED_MODEL

TRITON_URL = os.environ.get("TRITON_URL")
TRITON_MAX_BATCH = 64


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TritonEmbedder:
    def __init__(self, url: str, model: str = "bge_small"):
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.client = httpx.Client(base_url=url, timeout=30)
        self.path = f"/v2/models/{model}/infer"

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        if len(texts) > TRITON_MAX_BATCH:
            return np.concatenate([self.encode(texts[i:i + TRITON_MAX_BATCH])
                                   for i in range(0, len(texts), TRITON_MAX_BATCH)])
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
        inputs = [{"name": k, "shape": list(enc[k].shape), "datatype": "INT64", "data": enc[k].astype(np.int64).ravel().tolist()}
                  for k in ("input_ids", "attention_mask")]
        out = self.client.post(self.path, json={"inputs": inputs}).raise_for_status().json()["outputs"][0]
        return np.asarray(out["data"], dtype=np.float32).reshape(out["shape"])


def load_embedder():
    return TritonEmbedder(TRITON_URL) if TRITON_URL else SentenceTransformer(EMBED_MODEL, device=pick_device())
