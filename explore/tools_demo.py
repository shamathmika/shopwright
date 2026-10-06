from shopwright.tools import run_tool

print(run_tool("filter_products", '{"kind": "computers", "max_price": 500, "k": 2}')[:300], "\n")
print(run_tool("search_products", '{"q": "earbuds"}'), "\n")
print(run_tool("buy_now", "{}"), "\n")
print(run_tool("search_products", "{not json"), "\n")
print(run_tool("filter_products", '{"max_price": "fifty"}'))
