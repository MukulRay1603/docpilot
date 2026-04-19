import json
import re
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from config import MODEL_DIR, CORPUS_PATH
from serve.inference import XRQAEngine

engine: Optional[XRQAEngine] = None
corpus: list[dict] = []
bm25: Optional[BM25Okapi] = None


def _tokenize(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    return re.sub(r"[^\w\s]", " ", text).split()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, corpus, bm25
    engine = XRQAEngine(MODEL_DIR)
    if CORPUS_PATH.exists():
        corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        bm25 = BM25Okapi([_tokenize(doc["text"]) for doc in corpus])
        print(f"Loaded {len(corpus)} passages from {CORPUS_PATH.name}")
    yield
    engine = None


app = FastAPI(title="QA Engine", version="1.0.0", lifespan=lifespan)


class AnswerRequest(BaseModel):
    question: str
    context: str


class CorpusRequest(BaseModel):
    question: str
    top_k: int = 3


class AnswerResponse(BaseModel):
    answer: str
    score: float
    context_used: Optional[str] = None
    latency: dict


def _retrieve(question: str, top_k: int) -> list[str]:
    if bm25 is None:
        return []
    scores = bm25.get_scores(_tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [corpus[i]["text"] for i in top]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None, "corpus_size": len(corpus)}


@app.get("/metrics")
def metrics():
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    return engine.latency_stats()


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    result = engine.answer(req.question, req.context)
    return AnswerResponse(answer=result["answer"], score=result["score"], latency=result["latency"])


@app.post("/answer/corpus", response_model=AnswerResponse)
def answer_from_corpus(req: CorpusRequest):
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    if not corpus:
        raise HTTPException(503, "Corpus not loaded")
    passages = _retrieve(req.question, req.top_k)
    context = " ".join(passages)
    result = engine.answer(req.question, context)
    return AnswerResponse(
        answer=result["answer"],
        score=result["score"],
        context_used=context[:300] + "…" if len(context) > 300 else context,
        latency=result["latency"],
    )
