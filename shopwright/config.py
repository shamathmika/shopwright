import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
KINDS = {"headphones", "speakers", "cameras", "computers"}

load_dotenv(ROOT / ".env")
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_EXTRA_BODY = json.loads(os.environ["LLM_EXTRA_BODY"])
RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL")
