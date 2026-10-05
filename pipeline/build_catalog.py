import orjson
import pandas as pd
from shopwright.config import DATA

META = DATA / "meta_Electronics.jsonl"
OUT = DATA / "catalog.parquet"

SLICES = {
    "Headphones, Earbuds & Accessories > Headphones & Earbuds": "headphones",
    "Portable Audio & Video > Portable Speakers & Docks": "speakers",
    "Camera & Photo > Digital Cameras": "cameras",
    "Computers & Accessories > Computers & Tablets": "computers",
}
MIN_RATINGS = 5

def parse_price(v):
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except ValueError:
        return None

rows = []
with open(META, "rb") as f:
    for line in f:
        p = orjson.loads(line)
        kind = SLICES.get(" > ".join((p.get("categories") or [])[1:3]))
        if kind is None or (p.get("rating_number") or 0) < MIN_RATINGS:
            continue
        price = parse_price(p.get("price"))
        if price is None or price < 1:
            continue
        d = p.get("details") or {}
        rows.append({
            "parent_asin": p["parent_asin"],
            "kind": kind,
            "title": p.get("title") or "",
            "brand": d.get("Brand") or p.get("store") or "",
            "price": price,
            "rating": p.get("average_rating"),
            "n_ratings": p["rating_number"],
            "features": p.get("features") or [],
            "description": " ".join(p.get("description") or []),
            "details": orjson.dumps(d).decode(),
        })

df = pd.DataFrame(rows)
df = (df.sort_values("n_ratings", ascending=False)
        .drop_duplicates("title")
        .sort_index()
        .reset_index(drop=True))

df.to_parquet(OUT, index=False)
print(df.shape)
print(df["kind"].value_counts())
print(df["price"].describe())
