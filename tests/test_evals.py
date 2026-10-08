import json

import pandas as pd

from agreement import cohen_kappa
from checks import acknowledges_none, filter_args_ok, guessed_ids
from gate import check


def test_acknowledges_none():
    assert acknowledges_none("Unfortunately we do not carry drones.")
    assert acknowledges_none("I couldn't find any Pyle laptops, but here are tablets")
    assert not acknowledges_none("Here are some Pyle laptops")


def test_filter_args_ok():
    c = {"kind": "computers", "brand": "HP", "max_price": 500}
    assert filter_args_ok({"kind": "computers", "brand": "hp", "max_price": 500}, c)
    assert not filter_args_ok({"kind": "computers", "max_price": 500}, c)
    assert not filter_args_ok({"kind": "computers", "brand": "HP", "max_price": 800}, c)


def test_guessed_ids_counts_ids_never_returned_by_a_tool():
    trace = [
        {"tool": "search_products", "args": "{}", "error": False, "asins": ["B0AAAAAAAA"]},
        {"tool": "review_summary", "args": json.dumps({"asin": "B0AAAAAAAA"}), "error": False, "asins": []},
        {"tool": "compare_products", "args": json.dumps({"asins": ["B0AAAAAAAA", "B0EXAMPLE1"]}), "error": True, "asins": []},
    ]
    assert guessed_ids(trace) == 1


def test_cohen_kappa():
    a = pd.Series([True, False, True, False])
    assert cohen_kappa(a, a) == 1.0


def summary(success, test, invented=0.0, sha="x"):
    return {"eval_set_sha": sha, "overall": {"success": success, "invented": invented},
            "by_split": {"test": {"success": test}}}


CFG = {"min_success": 0.65, "max_success_drop": 0.03, "max_test_success_drop": 0.03, "max_invented_increase": 0.01}


def test_gate_passes_identical_runs():
    assert check(summary(0.72, 0.74), summary(0.72, 0.74), CFG) == []


def test_gate_fails_on_regression():
    failures = check(summary(0.72, 0.74), summary(0.52, 0.53), CFG)
    assert any("success dropped" in f for f in failures)
    assert any("below minimum" in f for f in failures)


def test_gate_fails_on_different_eval_sets():
    assert check(summary(0.72, 0.74, sha="a"), summary(0.72, 0.74, sha="b"), CFG)
