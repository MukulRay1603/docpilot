"""UI checks ONLY. Run from the repo root:  python docs/ui/standin_server.py

Serves the real FastAPI app with a STAND-IN QA engine that returns the first
sentence of the retrieved context. Use it to look at the interface when the
real model isn't available. Its answers and scores are NOT model output and
must never be reported as metrics. Set NO_MODEL=1 to see the "model not
loaded" state instead.
"""
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DOCPILOT_BM25_ONLY", "1")

import uvicorn  # noqa: E402

import serve.app as app_module  # noqa: E402
from serve.retrieval import Retriever  # noqa: E402


class StandIn:
    label = "STAND-IN for UI checks only (not the real model)"

    def answer(self, question, context):
        first = re.split(r"(?<=[.!?])\s+", context.strip())[0] if context.strip() else ""
        return {"answer": first, "score": 1.0, "latency": {}}

    def latency_stats(self):
        return {"note": "stand-in engine"}


@asynccontextmanager
async def lifespan(app):
    if os.getenv("NO_MODEL") != "1":
        app_module.engine = StandIn()
    corpus_path = Path(os.getenv("CORPUS", ROOT / "data" / "corpus.json"))
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    app_module.corpus, app_module.retriever = data, Retriever(data)
    yield


app_module.app.router.lifespan_context = lifespan

if __name__ == "__main__":
    uvicorn.run(app_module.app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
