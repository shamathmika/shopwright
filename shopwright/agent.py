import json
import operator
import re
import time
from functools import cache
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from shopwright.config import LLM_API_KEY, LLM_BASE_URL, LLM_EXTRA_BODY, LLM_MODEL
from shopwright.tools import TOOLS, call_tool

SYSTEM_PROMPT = """You are a shopping assistant for an electronics store selling headphones, speakers, cameras and computers.
Always use the tools to find products. Never invent products, prices or specs.
Only recommend products that appear in tool results, and state their price.
If a tool returns an error, fix your call and try again.
If nothing matches, say so and suggest loosening a constraint.
Keep answers short.
When you need the same tool for several products, request all of those calls in one turn.
Check reviews for at most 3 products.
Every time you mention a product, write its asin from the tool results in parentheses right after its name, in this format: Product Name (B0EXAMPLE1). B0EXAMPLE1 is only a format example, never a real asin.
If the user names a specific product, find it with search_products first. Never guess an asin.
If the answer depends on what reviewers say (for example "best reviews for", "complaints", "what people think"), call review_summary. Do not answer from star ratings alone.
"""

MAX_STEPS = 8
FINALIZE_PROMPT = "Step limit reached. Answer now using only the tool results above."
ASIN_IN_RESULT = re.compile(r'"asin": "([^"]+)"')
MAX_TRACE_RESULT_CHARS = 4000


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    trace: Annotated[list[dict], operator.add]
    hit_limit: bool


@cache
def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
                      temperature=0, extra_body=LLM_EXTRA_BODY)


def agent_node(state: AgentState) -> dict:
    reply = get_chat_model().bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [reply], "steps": state["steps"] + 1}


def tools_node(state: AgentState) -> dict:
    last: AIMessage = state["messages"][-1]
    calls = [(c["id"], c["name"], json.dumps(c["args"])) for c in last.tool_calls]
    calls += [(c["id"], c["name"], c["args"] or "") for c in last.invalid_tool_calls]
    results, trace = [], []
    for call_id, name, args in calls:
        t0 = time.perf_counter()
        content = call_tool(name, args)
        trace.append({"step": state["steps"], "tool": name, "args": args,
                      "error": content.startswith('{"error"'),
                      "asins": list(dict.fromkeys(ASIN_IN_RESULT.findall(content))),
                      "result": content[:MAX_TRACE_RESULT_CHARS],
                      "ms": round((time.perf_counter() - t0) * 1000)})
        results.append(ToolMessage(content=content, tool_call_id=call_id))
    return {"messages": results, "trace": trace}


def finalize_node(state: AgentState) -> dict:
    nudge = HumanMessage(FINALIZE_PROMPT)
    reply = get_chat_model().invoke(state["messages"] + [nudge])
    return {"messages": [nudge, reply], "hit_limit": True}


def after_agent(state: AgentState) -> str:
    last: AIMessage = state["messages"][-1]
    return "tools" if last.tool_calls or last.invalid_tool_calls else END


def after_tools(state: AgentState) -> str:
    return "agent" if state["steps"] < MAX_STEPS else "finalize"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()


AGENT = build_graph()


def initial_state(question: str) -> AgentState:
    return {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)],
            "steps": 0, "trace": [], "hit_limit": False}


def run_agent(question: str) -> dict:
    final = AGENT.invoke(initial_state(question), {"recursion_limit": 2 * MAX_STEPS + 5})
    return {"answer": final["messages"][-1].content, "steps": final["steps"],
            "trace": final["trace"], "hit_limit": final["hit_limit"]}
