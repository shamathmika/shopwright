import json
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

from shopwright.config import DATA

OUT = Path(__file__).resolve().parent / "data" / "eval_set.jsonl"
SEED = 7
MODEL_CODE = r"[A-Za-z]+-?\d|\d+-?[A-Za-z]"

COUNTS = {"constraint": 60, "lookup": 30, "compare": 30, "reviews": 30,
          "multi": 15, "out_of_scope": 20, "impossible": 15}
EXPECTED_STEPS = {"constraint": 2, "lookup": 2, "compare": 3, "reviews": 3,
                  "multi": 4, "out_of_scope": 1, "impossible": 2}
KIND_WORDS = {"headphones": "headphones", "speakers": "Bluetooth speaker",
              "cameras": "digital camera", "computers": "laptop"}
KIND_PLURALS = {"headphones": "headphones", "speakers": "Bluetooth speakers",
                "cameras": "digital cameras", "computers": "laptops"}
TITLE_PATTERNS = {"computers": "laptop|notebook|chromebook|macbook"}
PRICES = [25, 30, 50, 75, 100, 150, 200, 300, 400, 500, 700, 1000]
ASPECTS = {"headphones": ["comfort", "battery life", "sound quality", "noise cancelling"],
           "speakers": ["bass", "battery life", "waterproofing", "sound quality"],
           "cameras": ["image quality", "battery life", "autofocus", "ease of use"],
           "computers": ["battery life", "keyboard", "screen", "performance"]}
OUT_OF_SCOPE = ["refrigerator", "washing machine", "running shoes", "sofa", "coffee maker",
                "dog food", "lipstick", "car tires", "microwave", "yoga mat", "electric guitar",
                "smartwatch", "printer", "drone", "4K TV", "gaming console", "robot vacuum",
                "lawn mower", "perfume", "backpack"]


def load() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(DATA / "catalog.parquet")
    df["model"] = df["details"].map(lambda d: orjson.loads(d).get("Model Name"))
    n_reviews = pd.read_parquet(DATA / "reviews.parquet", columns=["parent_asin"])["parent_asin"].value_counts()
    df["n_reviews"] = df["parent_asin"].map(n_reviews).fillna(0).astype(int)
    return df, n_reviews


def matches(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    m = df["kind"] == c["kind"]
    if c.get("brand"):
        m &= df["brand"].str.contains(c["brand"], case=False, regex=False)
    if c.get("max_price") is not None:
        m &= df["price"] <= c["max_price"]
    if c.get("min_price") is not None:
        m &= df["price"] >= c["min_price"]
    if c.get("min_rating") is not None:
        m &= df["rating"] >= c["min_rating"]
    if c.get("title_pattern"):
        m &= df["title"].str.contains(c["title_pattern"], case=False, regex=True)
    return df[m]


def distinctive(df: pd.DataFrame) -> pd.DataFrame:
    named = df[df["model"].notna()].copy()
    named["model"] = named["model"].str.strip()
    named = named[named["model"].str.len().between(3, 30) & named["model"].str.contains(MODEL_CODE)]
    named = named[named["model"].str.lower() != named["brand"].str.lower()]
    titles = df["title"].str.lower()
    unique = named["model"].map(lambda m: titles.str.contains(m.lower(), regex=False).sum() == 1)
    return named[unique]


def name_of(row) -> str:
    model = row["model"]
    return model if row["brand"].lower() in model.lower() else f"{row['brand']} {model}"


def constraint_items(df, rng, n, category):
    items, seen = [], set()
    templates = ["Find me {brand}{plural} under ${p}.",
                 "I need {word} below ${p} with at least {r} stars.",
                 "Show me {brand}{plural} between ${lo} and ${p}.",
                 "What {brand}{plural} can I get for under ${p}?"]
    while len(items) < n:
        kind = rng.choice(sorted(KIND_WORDS))
        brands = df[df["kind"] == kind]["brand"].value_counts().head(6).index.tolist()
        brand = rng.choice(brands) if rng.random() < 0.6 else None
        t = int(rng.integers(len(templates)))
        prices = df[df["kind"] == kind]["price"]
        allowed = [x for x in PRICES if prices.quantile(0.25) <= x <= prices.quantile(0.9)]
        p = int(rng.choice(allowed))
        c = {"kind": kind, "brand": brand, "max_price": p,
             "title_pattern": TITLE_PATTERNS.get(kind)}
        if t == 1:
            c["brand"], c["min_rating"] = None, float(rng.choice([4.0, 4.5]))
        if t == 2:
            c["min_price"] = int(p * 0.5)
        if category == "multi":
            c.pop("min_rating", None)
            c.pop("min_price", None)
            c["brand"] = c["brand"] or brands[0]
        key = json.dumps(c, sort_keys=True)
        if key in seen or len(matches(df, c)) < 3:
            continue
        seen.add(key)
        brand_txt = f"{c['brand']} " if c.get("brand") else ""
        if category == "multi":
            aspect = str(rng.choice(ASPECTS[kind]))
            q = f"Which {brand_txt}{KIND_WORDS[kind]} under ${p} has the best reviews for {aspect}?"
            items.append({"category": "multi", "question": q,
                          "expected": {"constraints": c, "aspect": aspect}})
            continue
        word = KIND_WORDS[kind] if kind == "headphones" else f"a {KIND_WORDS[kind]}"
        q = templates[t].format(brand=brand_txt, word=word, plural=KIND_PLURALS[kind], p=p,
                                r=c.get("min_rating"), lo=c.get("min_price"))
        items.append({"category": "constraint", "question": q, "expected": {"constraints": c}})
    return items


def lookup_items(pool, rng, n):
    templates = ["Tell me about the {name}.", "How much is the {name}?", "Do you have the {name}?"]
    picked = pool.sample(n, random_state=SEED)
    return [{"category": "lookup", "question": templates[i % 3].format(name=name_of(r)),
             "expected": {"target": r["parent_asin"]}} for i, (_, r) in enumerate(picked.iterrows())]


def compare_items(pool, rng, n):
    popular = pool[pool["n_ratings"] >= 200]
    items, used = [], set()
    templates = ["Compare the {a} and the {b}.", "{a} vs {b}: which one should I buy?"]
    for kind, group in popular.groupby("kind"):
        rows = group.sample(frac=1, random_state=SEED)
        for i in range(0, len(rows) - 1, 2):
            a, b = rows.iloc[i], rows.iloc[i + 1]
            items.append({"category": "compare",
                          "question": templates[len(items) % 2].format(a=name_of(a), b=name_of(b)),
                          "expected": {"targets": [a["parent_asin"], b["parent_asin"]]}})
    order = rng.permutation(len(items))
    return [items[i] for i in order[:n]]


def review_items(pool, rng, n):
    rich = pool[pool["n_reviews"] >= 15].sample(n, random_state=SEED + 1)
    templates = ["What do customers say about the {aspect} of the {name}?",
                 "Do people complain about {aspect} on the {name}?"]
    items = []
    for i, (_, r) in enumerate(rich.iterrows()):
        aspect = str(rng.choice(ASPECTS[r["kind"]]))
        items.append({"category": "reviews",
                      "question": templates[i % 2].format(aspect=aspect, name=name_of(r)),
                      "expected": {"target": r["parent_asin"], "aspect": aspect}})
    return items


def out_of_scope_items():
    templates = ["Do you sell a {x}?", "Recommend a good {x} under $200.", "I'm looking for a {x}."]
    return [{"category": "out_of_scope", "question": templates[i % 3].format(x=x), "expected": {}}
            for i, x in enumerate(OUT_OF_SCOPE)]


def impossible_items(df, rng, n):
    items, seen = [], set()
    kinds = sorted(KIND_WORDS)
    all_brands = [b for k in kinds for b in df[df["kind"] == k]["brand"].value_counts().head(5).index]
    while len(items) < n:
        kind = str(rng.choice(kinds))
        if rng.random() < 0.5:
            c = {"kind": kind, "brand": str(rng.choice(all_brands)), "max_price": None,
                 "title_pattern": TITLE_PATTERNS.get(kind)}
            q = f"Show me {c['brand']} {KIND_PLURALS[kind]}."
        else:
            floor = df[df["kind"] == kind]["price"].min()
            p = max(1, int(floor) - 1)
            c = {"kind": kind, "brand": None, "max_price": p, "title_pattern": TITLE_PATTERNS.get(kind),
                 "min_rating": 4.5}
            q = f"I want {KIND_PLURALS[kind]} under ${p} with at least 4.5 stars."
        key = json.dumps(c, sort_keys=True)
        if key in seen or len(matches(df, c)) > 0:
            continue
        seen.add(key)
        items.append({"category": "impossible", "question": q, "expected": {"constraints": c}})
    return items


def main():
    rng = np.random.default_rng(SEED)
    df, _ = load()
    pool = distinctive(df)
    items = (constraint_items(df, rng, COUNTS["constraint"], "constraint")
             + lookup_items(pool, rng, COUNTS["lookup"])
             + compare_items(pool, rng, COUNTS["compare"])
             + review_items(pool, rng, COUNTS["reviews"])
             + constraint_items(df, rng, COUNTS["multi"], "multi")
             + out_of_scope_items()
             + impossible_items(df, rng, COUNTS["impossible"]))
    split_rng = np.random.default_rng(SEED + 100)
    splits = {}
    for cat in COUNTS:
        idx = [i for i, it in enumerate(items) if it["category"] == cat]
        order = split_rng.permutation(len(idx))
        for rank, j in enumerate(order):
            splits[idx[j]] = "dev" if rank < len(idx) // 2 else "test"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for i, item in enumerate(items):
            item = {"id": f"q{i:03d}", "split": splits[i], **item, "expected_steps": EXPECTED_STEPS[item["category"]]}
            f.write(json.dumps(item) + "\n")
    counts = pd.Series([it["category"] for it in items]).value_counts()
    print(f"{len(items)} items -> {OUT}\n{counts.to_string()}\ndistinctive products: {len(pool)}")


if __name__ == "__main__":
    main()
