import os
import sys
from pathlib import Path

os.environ.setdefault("LLM_BASE_URL", "http://localhost:0/v1")
os.environ.setdefault("LLM_API_KEY", "test")
os.environ.setdefault("LLM_MODEL", "test")
os.environ.setdefault("LLM_EXTRA_BODY", "{}")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
