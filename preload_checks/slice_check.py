from collections import Counter, defaultdict
import orjson

META = "data/meta_Electronics.jsonl"
SLICES = {
    "Headphones, Earbuds & Accessories > Headphones & Earbuds",
    "Portable Audio & Video > Portable Speakers & Docks",
    "Camera & Photo > Digital Cameras",
    "Computers & Accessories > Computers & Tablets",
}
MIN_RATINGS = 20

usable = Counter()
specs = defaultdict(Counter)

with open(META, "rb") as f:
    for line in f:
        p = orjson.loads(line)
        cat = " > ".join((p.get("categories") or [])[1:3])
        if cat not in SLICES:
            continue
        if p.get("price") is None or (p.get("rating_number") or 0) < MIN_RATINGS:
            continue
        usable[cat] += 1
        specs[cat].update((p.get("details") or {}).keys())

for cat in SLICES:
    n = usable[cat]
    print(f"\n{n:,} usable  {cat}")
    for k, v in specs[cat].most_common(12):
        print(f"  {v/n:>4.0%}  {k}")
