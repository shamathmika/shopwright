import orjson

SKIP = {
    "Best Sellers Rank", "Date First Available", "Is Discontinued By Manufacturer",
    "Manufacturer", "Brand", "Package Dimensions", "Item Dimensions  LxWxH",
    "Country of Origin", "Number Of Items",
}
MAX_FEATURES = 5


def product_text(row) -> str:
    details = orjson.loads(row["details"])
    specs = "; ".join(f"{k}: {v}" for k, v in details.items()
                      if k not in SKIP and isinstance(v, str))
    parts = [row["title"], f"Brand: {row['brand']}. Type: {row['kind']}.",
             specs, *row["features"][:MAX_FEATURES]]
    return "\n".join(p for p in parts if p)
