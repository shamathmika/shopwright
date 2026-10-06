import json
import re
from functools import cache

import pandas as pd

from shopwright.config import DATA

ASIN_IN_TEXT = re.compile(r"\bB0[A-Z0-9]{8}\b")
PRICE_TOLERANCE = 0.01
NONE_FOUND = re.compile(r"\b(no|not|none|couldn't|could not|don't|do not|doesn't|unable|unfortunately)\b[^.]{0,60}\b(find|found|have|carry|sell|available|match|matches|offer|stock)", re.IGNORECASE)


@cache
def catalog() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "catalog.parquet").set_index("parent_asin")
    df["model"] = df["details"].map(lambda d: json.loads(d).get("Model Name") or "")
    return df


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def retrieved_asins(trace: list[dict]) -> set[str]:
    return {a for t in trace if not t["error"] for a in t.get("asins", [])}


def mentioned_asins(answer: str, trace: list[dict]) -> set[str]:
    cat = catalog()
    found = {a for a in ASIN_IN_TEXT.findall(answer or "") if a in cat.index}
    text = " ".join(normalize(answer or "").split())
    for asin in retrieved_asins(trace) - found:
        if asin in cat.index:
            prefix = " ".join(normalize(cat.at[asin, "title"]).split()[:5])
            model = " ".join(normalize(cat.at[asin, "model"]).split())
            if (len(prefix) >= 15 and prefix in text) or (len(model) >= 4 and f" {model} " in f" {text} "):
                found.add(asin)
    return found


def invented_asins(answer: str, trace: list[dict]) -> set[str]:
    return set(ASIN_IN_TEXT.findall(answer or "")) - retrieved_asins(trace)


def acknowledges_none(answer: str) -> bool:
    return bool(NONE_FOUND.search(answer or ""))


def guessed_ids(trace: list[dict]) -> int:
    seen, guessed = set(), 0
    for t in trace:
        try:
            args = json.loads(t["args"]) if isinstance(t["args"], str) else t["args"]
        except json.JSONDecodeError:
            args = {}
        ids = args.get("asins", []) if t["tool"] == "compare_products" else [args.get("asin")] if t["tool"] == "review_summary" else []
        ids = [ids] if isinstance(ids, str) else ids
        guessed += sum(1 for i in ids if i and i not in seen)
        if not t["error"]:
            seen.update(t.get("asins", []))
    return guessed


def satisfies(asin: str, c: dict) -> bool:
    cat = catalog()
    if asin not in cat.index:
        return False
    row = cat.loc[asin]
    if row["kind"] != c["kind"]:
        return False
    if c.get("brand") and c["brand"].lower() not in row["brand"].lower():
        return False
    if c.get("max_price") is not None and row["price"] > c["max_price"]:
        return False
    if c.get("min_price") is not None and row["price"] < c["min_price"]:
        return False
    if c.get("min_rating") is not None and (pd.isna(row["rating"]) or row["rating"] < c["min_rating"]):
        return False
    if c.get("title_pattern") and not re.search(c["title_pattern"], row["title"], re.IGNORECASE):
        return False
    return True


def calls(trace: list[dict], tool: str) -> list[dict]:
    out = []
    for t in trace:
        if t["tool"] != tool or t["error"]:
            continue
        try:
            out.append(json.loads(t["args"]) if isinstance(t["args"], str) else t["args"])
        except json.JSONDecodeError:
            continue
    return out


def price_ok(got, want) -> bool:
    if want is None:
        return True
    return got is not None and abs(float(got) - want) <= max(1.0, want * PRICE_TOLERANCE)


def filter_args_ok(args: dict, c: dict) -> bool:
    if args.get("kind") != c["kind"]:
        return False
    if c.get("brand") and c["brand"].lower() not in str(args.get("brand", "")).lower():
        return False
    if not price_ok(args.get("max_price"), c.get("max_price")):
        return False
    if c.get("min_price") is not None and not price_ok(args.get("min_price"), c["min_price"]):
        return False
    if c.get("min_rating") is not None and (args.get("min_rating") is None or float(args["min_rating"]) < c["min_rating"] - 0.01):
        return False
    return True


def grade(item: dict, result: dict) -> dict:
    cat, exp = item["category"], item["expected"]
    trace, answer = result["trace"], result["answer"] or ""
    rec = mentioned_asins(answer, trace)
    tools_used = {t["tool"] for t in trace}

    if cat in ("constraint", "impossible", "multi"):
        c = exp["constraints"]
        constraint_ok = all(satisfies(a, c) for a in rec)
        filter_ok = any(filter_args_ok(a, c) for a in calls(trace, "filter_products"))
        if cat == "constraint":
            success = bool(rec) and constraint_ok
            tool_ok = filter_ok
        elif cat == "impossible":
            success = not rec or acknowledges_none(answer)
            tool_ok = filter_ok or "search_products" in tools_used
        else:
            reviewed = {a["asin"] for a in calls(trace, "review_summary") if "asin" in a}
            success = bool(rec) and constraint_ok and any(satisfies(a, c) for a in reviewed)
            tool_ok = filter_ok and bool(reviewed)
    elif cat == "lookup":
        success = exp["target"] in rec
        tool_ok = exp["target"] in retrieved_asins(trace)
    elif cat == "compare":
        compared = [set(a.get("asins", [])) for a in calls(trace, "compare_products")]
        tool_ok = any(set(exp["targets"]) <= s for s in compared)
        success = tool_ok and set(exp["targets"]) <= rec
    elif cat == "reviews":
        reviewed = {a.get("asin") for a in calls(trace, "review_summary")}
        tool_ok = exp["target"] in reviewed
        success = tool_ok and exp["target"] in rec
    elif cat == "out_of_scope":
        success = not rec or acknowledges_none(answer)
        tool_ok = tools_used <= {"search_products"}
    else:
        raise ValueError(f"unknown category {cat}")

    extra = max(0, result["steps"] - item["expected_steps"])
    return {
        "success": bool(success),
        "tool_correct": bool(tool_ok),
        "steps": result["steps"],
        "efficiency": max(0.0, 1 - 0.25 * extra),
        "tool_errors": sum(t["error"] for t in trace),
        "hit_limit": result["hit_limit"],
        "recommended": sorted(rec),
        "invented_asins": sorted(invented_asins(answer, trace)),
        "guessed_ids": guessed_ids(trace),
    }
