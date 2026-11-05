"""UI-check launcher ONLY. Real QA model can't be downloaded here, so this
swaps in a stand-in engine that returns the first sentence of the context.
Its answers and scores are NOT model output and must not be reported as metrics."""
import json, os, sys, re
from contextlib import asynccontextmanager
sys.path.insert(0, "/home/user/docpilot")
os.environ.setdefault("DOCPILOT_BM25_ONLY", "1")
import serve.app as A
from serve.retrieval import Retriever

class StandIn:
    label = "STAND-IN for UI checks only (not the real model)"
    def answer(self, q, ctx):
        s = re.split(r"(?<=[.!?])\s+", ctx.strip())[0] if ctx.strip() else ""
        return {"answer": s, "score": 1.0, "latency": {}}
    def latency_stats(self): return {"note": "stand-in engine"}

@asynccontextmanager
async def lifespan(app):
    if os.getenv("NO_MODEL") != "1":
        A.engine = StandIn()
    data = json.load(open(os.environ.get("CORPUS", "/tmp/claude-0/demo/corpus.json")))
    A.corpus, A.retriever = data, Retriever(data)
    yield
A.app.router.lifespan_context = lifespan
import uvicorn
uvicorn.run(A.app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")), log_level="warning")
