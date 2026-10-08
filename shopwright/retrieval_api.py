from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool

from shopwright.resources import get_reviews, get_searcher
from shopwright.tools import run_tool


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_searcher()
    get_reviews()
    yield


app = FastAPI(title="ShopWright retrieval", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/tools/{name}")
async def tool(name: str, request: Request) -> Response:
    arguments = (await request.body()).decode()
    return Response(await run_in_threadpool(run_tool, name, arguments), media_type="application/json")
