from functools import cache

import pandas as pd
from openai import OpenAI

from shopwright.config import DATA, LLM_API_KEY, LLM_BASE_URL
from shopwright.search import Searcher


@cache
def get_searcher() -> Searcher:
    return Searcher()


@cache
def get_reviews() -> dict[str, pd.DataFrame]:
    return dict(tuple(pd.read_parquet(DATA / "reviews.parquet").groupby("parent_asin")))


@cache
def get_llm() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
