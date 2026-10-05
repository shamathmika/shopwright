import orjson
from shopwright.config import DATA

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

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_parquet(DATA / "catalog.parquet")
    texts = df.apply(product_text, axis=1)
    print(texts.str.len().describe())
    for kind in df["kind"].unique():
        print(f"\n===== {kind} =====\n{texts[df['kind'] == kind].iloc[0]}")
