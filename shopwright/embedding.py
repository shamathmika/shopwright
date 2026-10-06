import torch
from sentence_transformers import SentenceTransformer

from shopwright.config import EMBED_MODEL


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL, device=pick_device())
