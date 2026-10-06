import re

from shopwright.config import LLM_EXTRA_BODY, LLM_MODEL
from shopwright.resources import get_llm, get_reviews, get_searcher
from shopwright.tools.catalog import product_card

SUMMARY_PROMPT = (
    "You summarize customer reviews for a shopper. Use ONLY the reviews provided. "
    "Write at most 5 short bullets: main pros, main cons, and any repeated complaints. "
    "If a focus topic is given, answer about that topic first. Do not invent facts."
)
MIN_REVIEWS = 5
MAX_REVIEWS = 15


def review_summary(asin: str, focus: str | None = None) -> dict:
    df = get_searcher().catalog
    match = df[df["parent_asin"] == asin]
    if match.empty:
        return {"error": f"unknown asin {asin!r}; use an asin from search or filter results"}
    card = product_card(match.iloc[0])
    sub = get_reviews().get(asin)
    n = 0 if sub is None else len(sub)
    if n < MIN_REVIEWS:
        return {"product": card, "reviews_available": n, "summary": None,
                "note": "too few written reviews to summarize"}

    stars = {int(k): int(v) for k, v in sub["rating"].value_counts().sort_index().items()}
    sub = sub.assign(match=False)
    if focus:
        words = [re.escape(w) for w in focus.lower().split() if len(w) > 2]
        if words:
            sub["match"] = sub["text"].str.lower().str.contains("|".join(words))
    picked = sub.sort_values(["match", "helpful_vote"], ascending=False).head(MAX_REVIEWS)

    reviews_text = "\n\n".join(f"[{int(r.rating)} stars] {r.title}: {r.text.replace('<br />', ' ')[:600]}"
                               for r in picked.itertuples())
    user = f"Product: {card['title']}\nFocus: {focus or 'none'}\n\nReviews:\n{reviews_text}"
    resp = get_llm().chat.completions.create(
        model=LLM_MODEL, temperature=0.2, max_tokens=300, extra_body=LLM_EXTRA_BODY,
        messages=[{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": user}])

    choice = resp.choices[0]
    if not choice.message.content:
        return {"error": f"summary generation failed (finish_reason={choice.finish_reason})"}
    return {"product": card, "reviews_available": n, "reviews_used": len(picked),
            "star_counts": stars, "summary": choice.message.content}
