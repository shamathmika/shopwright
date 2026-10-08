import json

from shopwright.tools import TOOLS, run_tool
from shopwright.tools.registry import TOOL_FUNCS


def test_every_tool_schema_is_complete():
    for tool in TOOLS:
        fn = tool["function"]
        assert set(fn) == {"name", "description", "parameters"}, fn.keys()
        assert fn["parameters"]["type"] == "object"
        assert fn["parameters"]["properties"]
        assert set(fn["parameters"]["required"]) <= set(fn["parameters"]["properties"])


def test_every_schema_has_a_function_and_vice_versa():
    assert {t["function"]["name"] for t in TOOLS} == set(TOOL_FUNCS)


def test_unknown_tool_returns_error():
    out = json.loads(run_tool("buy_now", "{}"))
    assert "unknown tool" in out["error"]


def test_broken_json_returns_error():
    out = json.loads(run_tool("search_products", "{not json"))
    assert "not valid JSON" in out["error"]
