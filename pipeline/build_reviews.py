import heapq

import orjson
import pandas as pd

from shopwright.config import DATA

REVIEWS = DATA / "Electronics.jsonl"
CATALOG = DATA / "catalog.parquet"
OUT = DATA / "reviews.parquet"
PER_PRODUCT = 30

wanted = set(pd.read_parquet(CATALOG, columns=["parent_asin"])["parent_asin"])
heaps = {}   # parent_asin -> min-heap of (helpful_vote, timestamp, seq, row)
seq = 0

with open(REVIEWS, "rb") as f:
    for n, line in enumerate(f, 1):
        r = orjson.loads(line)
        asin = r["parent_asin"]
        if asin not in wanted:
            continue
        row = {
            "parent_asin": asin,
            "rating": r["rating"],
            "title": r.get("title") or "",
            "text": r.get("text") or "",
            "helpful_vote": r.get("helpful_vote") or 0,
            "verified": r.get("verified_purchase", False),
            "timestamp": r["timestamp"],
        }
        item = (row["helpful_vote"], row["timestamp"], seq, row)
        seq += 1
        h = heaps.setdefault(asin, [])
        if len(h) < PER_PRODUCT:
            heapq.heappush(h, item)
        else:
            heapq.heappushpop(h, item)
        if n % 5_000_000 == 0:
            print(f"{n:,} read, {len(heaps):,} products hit", flush=True)

df = pd.DataFrame([it[3] for h in heaps.values() for it in h])
df.to_parquet(OUT, index=False)
print(df.shape)
print(df.groupby("parent_asin").size().describe())
