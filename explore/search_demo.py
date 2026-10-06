from shopwright.search import Searcher

s = Searcher()
for q in ["GWTN156", "noise cancelling earbuds for running",
          "something to take photos on vacation", "laptop under $500"]:
    print(f"\n=== {q}")
    for h in s.search(q, k=5):
        r = s.catalog.iloc[h.row]
        print(f"  dense={h.dense_rank!s:>4} bm25={h.bm25_rank!s:>4}  ${r.price:>8.2f}  {r.title[:65]}")
