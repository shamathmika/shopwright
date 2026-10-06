import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from shopwright.agent import AGENT, MAX_STEPS, initial_state, run_agent
from shopwright.resources import get_reviews, get_searcher

ANSWER_NODES = {"agent", "finalize"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_searcher()
    get_reviews()
    yield


app = FastAPI(title="ShopWright", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    t0 = time.perf_counter()
    out = run_agent(req.question)
    return {**out, "latency_ms": round((time.perf_counter() - t0) * 1000)}


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_events(question: str) -> AsyncIterator[str]:
    t0 = time.perf_counter()
    answer, steps, hit_limit = "", 0, False
    config = {"recursion_limit": 2 * MAX_STEPS + 5}
    try:
        async for mode, payload in AGENT.astream(initial_state(question), config,
                                                  stream_mode=["messages", "updates"]):
            if mode == "messages":
                chunk, meta = payload
                if meta.get("langgraph_node") in ANSWER_NODES and chunk.content and not chunk.tool_call_chunks:
                    yield sse("token", {"text": chunk.content})
                continue
            for node, update in payload.items():
                if node == "agent":
                    steps = update["steps"]
                    msg = update["messages"][-1]
                    for call in msg.tool_calls:
                        yield sse("tool_call", {"step": steps, "tool": call["name"], "args": call["args"]})
                    if not msg.tool_calls and not msg.invalid_tool_calls:
                        answer = msg.content
                elif node == "tools":
                    for t in update["trace"]:
                        yield sse("tool_result", {k: t[k] for k in ("step", "tool", "error", "ms")})
                elif node == "finalize":
                    answer, hit_limit = update["messages"][-1].content, True
        yield sse("done", {"answer": answer, "steps": steps, "hit_limit": hit_limit,
                           "latency_ms": round((time.perf_counter() - t0) * 1000)})
    except Exception as e:
        yield sse("error", {"message": f"{type(e).__name__}: {e}"})


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(stream_events(req.question), media_type="text/event-stream", headers=SSE_HEADERS)
