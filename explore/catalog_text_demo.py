import pandas as pd

from shopwright.catalog_text import product_text
from shopwright.config import DATA

df = pd.read_parquet(DATA / "catalog.parquet")
texts = df.apply(product_text, axis=1)
print(texts.str.len().describe())
for kind in df["kind"].unique():
    print(f"\n===== {kind} =====\n{texts[df['kind'] == kind].iloc[0]}")
