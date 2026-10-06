from collections import Counter

import numpy as np
import orjson
import pandas as pd

from shopwright.catalog_text import SKIP
from shopwright.config import KINDS
from shopwright.resources import get_searcher


def product_card(row) -> dict:
    return {
        "asin": row["parent_asin"],
        "title": row["title"][:120],
        "brand": row["brand"],
        "kind": row["kind"],
        "price": round(float(row["price"]), 2),
        "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
        "n_ratings": int(row["n_ratings"]),
    }


def search_products(query: str, k: int = 5) -> list[dict]:
    s = get_searcher()
    k = max(1, min(int(k), 10))
    return [product_card(s.catalog.iloc[h.row]) for h in s.search(query, k=k)]


def filter_products(kind: str | None = None, min_price: float | None = None,
                    max_price: float | None = None, brand: str | None = None,
                    min_rating: float | None = None, query: str | None = None,
                    k: int = 5) -> dict:
    s = get_searcher()
    df = s.catalog
    if kind and kind not in KINDS:
        return {"error": f"unknown kind {kind!r}; use one of {sorted(KINDS)}"}

    mask = pd.Series(True, index=df.index)
    if kind:
        mask &= df["kind"] == kind
    if min_price is not None:
        mask &= df["price"] >= float(min_price)
    if max_price is not None:
        mask &= df["price"] <= float(max_price)
    if brand:
        mask &= df["brand"].str.contains(brand, case=False, regex=False)
    if min_rating is not None:
        mask &= df["rating"] >= float(min_rating)

    rows = np.flatnonzero(mask.to_numpy())
    k = max(1, min(int(k), 10))
    if query:
        scores = s.emb[rows] @ s.embed_query(query)
        top = rows[np.argsort(-scores)[:k]]
    else:
        top = df.iloc[rows].sort_values("n_ratings", ascending=False).index[:k]
    return {"total_matches": len(rows), "products": [product_card(df.iloc[i]) for i in top]}


def compare_products(asins: list[str]) -> dict:
    if isinstance(asins, str):
        asins = asins.split(",")
    asins = list(dict.fromkeys(a.strip() for a in asins))[:4]
    if len(asins) < 2:
        return {"error": "give 2 to 4 different asins to compare"}

    df = get_searcher().catalog
    found = df[df["parent_asin"].isin(asins)]
    rows = {r["parent_asin"]: r for _, r in found.iterrows()}
    missing = [a for a in asins if a not in rows]
    if missing:
        return {"error": f"unknown asins {missing}; use asins from search or filter results"}
    if found["kind"].nunique() > 1:
        return {"error": f"can only compare the same kind of product; got {sorted(found['kind'].unique())}"}

    specs = {a: {k: v for k, v in orjson.loads(rows[a]["details"]).items()
                 if k not in SKIP and isinstance(v, str)} for a in asins}
    counts = Counter(k for d in specs.values() for k in d)
    keys = sorted(k for k, c in counts.items() if c >= 2)

    products = []
    for a in asins:
        card = product_card(rows[a])
        card["specs"] = {k: specs[a].get(k, "n/a")[:80] for k in keys}
        card["highlights"] = [f[:150] for f in rows[a]["features"][:3]]
        products.append(card)
    return {"products": products}
