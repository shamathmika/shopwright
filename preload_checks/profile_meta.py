from collections import Counter
import orjson

META = "data/meta_Electronics.jsonl"

cats = Counter()
specs = Counter()
has_price = 0

n = bad = 0
with open(META, "rb") as f:
    for line in f:
        try:
            p = orjson.loads(line)
        except orjson.JSONDecodeError:
            bad += 1
            continue
        n += 1

        c = p.get("categories") or []
        cats[" > ".join(c[1:3]) or "(none)"] += 1
        specs.update((p.get("details") or {}).keys())

        if p.get("price") is not None:
            has_price += 1

        if n % 1_000_000 == 0:
            print(f"{n:,}", flush=True)
    
print(f"done: {n:,} records, {bad} bad lines")
print(F"has price: {has_price/n:.0%}")
for k, v in cats.most_common(25):
    print(f"{v:>8,} {k}")
print("--- spec fields ---")
for k, v in specs.most_common(30):
    print(f"{v/n:>4.0%} {k}")