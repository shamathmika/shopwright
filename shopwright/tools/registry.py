import json
from functools import cache

import httpx

from shopwright.config import KINDS, RETRIEVAL_URL
from shopwright.tools.catalog import compare_products, filter_products, search_products
from shopwright.tools.reviews import review_summary

TOOL_FUNCS = {
    "search_products": search_products,
    "filter_products": filter_products,
    "compare_products": compare_products,
    "review_summary": review_summary,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_products",
        "description": "Find products that meet hard constraints: product kind, price range, brand, minimum rating. "
                       "Use this whenever the shopper states a budget, brand or rating requirement. "
                       "Optionally ranks matches by a free-text query. Returns total_matches and product cards. "
                       "Kinds are broad (computers includes laptops, tablets, desktops and smart displays), "
                       "so always set query to the specific product and use, e.g. 'laptop for college'.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What the shopper wants, in plain words"},
            "k": {"type": "integer", "description": "Number of results, 1-10. Default 5."},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "filter_products",
        "description": "Find products that meet hard constraints: product kind, price range, brand, minimum rating. "
                       "Use this whenever the shopper states a budget, brand or rating requirement. "
                       "Optionally ranks matches by a free-text query. Returns total_matches and product cards. "
                       "Kinds are broad (computers includes laptops, tablets, desktops and smart displays), "
                       "so always set query to the specific product and use, e.g. 'laptop for college'.",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": sorted(KINDS)},
            "min_price": {"type": "number", "description": "Lowest price in USD"},
            "max_price": {"type": "number", "description": "Highest price in USD"},
            "brand": {"type": "string", "description": "Brand name, e.g. Sony"},
            "min_rating": {"type": "number", "description": "Minimum average star rating, 1-5"},
            "query": {"type": "string", "description": "Optional free text to rank the matches"},
            "k": {"type": "integer", "description": "Number of results, 1-10. Default 5."},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "compare_products",
        "description": "Compare 2 to 4 products of the same kind side by side, with specs and key features. "
                       "Only use asins that appeared in earlier search or filter results.",
        "parameters": {"type": "object", "properties": {
            "asins": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
        }, "required": ["asins"]},
    }},
    {"type": "function", "function": {
        "name": "review_summary",
        "description": "Summarize what customers say about ONE product: pros, cons, repeated complaints, plus star counts. "
                       "Use when the shopper asks about quality, reliability, comfort or other people's experience. "
                       "Optional focus narrows it to one topic, e.g. 'battery life'.",
        "parameters": {"type": "object", "properties": {
            "asin": {"type": "string"},
            "focus": {"type": "string", "description": "Optional topic to focus on"},
        }, "required": ["asin"]},
    }},
]


def run_tool(name: str, arguments: str) -> str:
    func = TOOL_FUNCS.get(name)
    if func is None:
        result = {"error": f"unknown tool {name!r}; available: {sorted(TOOL_FUNCS)}"}
    else:
        try:
            result = func(**json.loads(arguments or "{}"))
        except json.JSONDecodeError as e:
            result = {"error": f"arguments were not valid JSON: {e}"}
        except TypeError as e:
            result = {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:
            result = {"error": f"{name} failed: {type(e).__name__}: {e}"}
    return json.dumps(result)


@cache
def retrieval_client() -> httpx.Client:
    return httpx.Client(base_url=RETRIEVAL_URL, timeout=120)


def call_tool(name: str, arguments: str) -> str:
    if not RETRIEVAL_URL:
        return run_tool(name, arguments)
    try:
        r = retrieval_client().post(f"/tools/{name}", content=arguments or "{}")
    except httpx.HTTPError as e:
        return json.dumps({"error": f"retrieval service unreachable: {type(e).__name__}"})
    if r.status_code != 200:
        return json.dumps({"error": f"retrieval service returned HTTP {r.status_code}"})
    return r.text
