import time

from shopwright.config import LLM_EXTRA_BODY, LLM_MODEL
from shopwright.resources import get_llm
from shopwright.tools import TOOLS, run_tool

SYSTEM_PROMPT = """You are a shopping assistant for an electronics store selling headphones, speakers, cameras and computers.
Always use the tools to find products. Never invent products, prices or specs.
Only recommend products that appear in tool results, and state their price.
If a tool returns an error, fix your call and try again.
If nothing matches, say so and suggest loosening a constraint.
Keep answers short.
When you need the same tool for several products, request all of those calls in one turn.
Check reviews for at most 3 products.
"""

MAX_STEPS = 8


def run_agent(question: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    trace = []
    for step in range(1, MAX_STEPS + 1):
        resp = get_llm().chat.completions.create(model=LLM_MODEL, messages=messages, tools=TOOLS,
                                                 temperature=0, extra_body=LLM_EXTRA_BODY)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return {"answer": msg.content, "steps": step, "trace": trace, "hit_limit": False}
        for tc in msg.tool_calls:
            t0 = time.perf_counter()
            result = run_tool(tc.function.name, tc.function.arguments)
            trace.append({"step": step, "tool": tc.function.name, "args": tc.function.arguments,
                          "error": result.startswith('{"error"'),
                          "ms": round((time.perf_counter() - t0) * 1000)})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    messages.append({"role": "user", "content": "Step limit reached. Answer now using only the tool results above."})
    resp = get_llm().chat.completions.create(model=LLM_MODEL, messages=messages,
                                             temperature=0, extra_body=LLM_EXTRA_BODY)
    return {"answer": resp.choices[0].message.content, "steps": MAX_STEPS, "trace": trace, "hit_limit": True}
