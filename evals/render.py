import json

MAX_SUMMARY_CHARS = 900


def render_result(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:600]
    if "error" in data:
        return f"ERROR: {data['error']}"
    lines = []
    if "total_matches" in data:
        lines.append(f"total_matches: {data['total_matches']}")
    products = data if isinstance(data, list) else data.get("products", [])
    for p in products:
        lines.append(f"- {p['asin']} | {p['title'][:90]} | ${p['price']} | {p['rating']}★ ({p['n_ratings']} ratings)")
        for k, v in p.get("specs", {}).items():
            lines.append(f"    {k}: {v}")
        for h in p.get("highlights", []):
            lines.append(f"    * {h}")
    if "summary" in data:
        prod = data["product"]
        lines.append(f"- {prod['asin']} | {prod['title'][:90]} | ${prod['price']} | {prod['rating']}★")
        lines.append(f"  reviews_available: {data.get('reviews_available')}, star_counts: {data.get('star_counts')}")
        lines.append(f"  summary: {(data['summary'] or data.get('note', ''))[:MAX_SUMMARY_CHARS]}")
    return "\n".join(lines)


def render_transcript(row: dict) -> str:
    parts = [f"USER QUESTION:\n{row['question']}\n"]
    if not row["trace"]:
        parts.append("TOOL CALLS: none\n")
    for i, t in enumerate(row["trace"], 1):
        parts.append(f"TOOL CALL {i}: {t['tool']}({t['args']})\nRESULT:\n{render_result(t.get('result', ''))}\n")
    parts.append(f"ASSISTANT FINAL ANSWER:\n{row['answer'] or '(empty)'}")
    return "\n".join(parts)
