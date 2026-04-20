import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel

from config import MODEL_DIR, CORPUS_PATH, SCORE_THRESHOLD
from serve.inference import QAEngine
from serve.retrieval import Retriever

engine: Optional[QAEngine] = None
corpus: list[dict] = []
retriever: Optional[Retriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, corpus, retriever
    engine = QAEngine(MODEL_DIR)
    corpus, retriever = _load_corpus(CORPUS_PATH)
    yield
    engine = None


def _load_corpus(path: Path) -> tuple[list[dict], Retriever]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"Loaded {len(data)} passages from {path.name}")
        return data, Retriever(data)
    print(f"No corpus at {path} -- starting empty")
    return [], Retriever([])


app = FastAPI(title="DocPilot", version="2.0.0", lifespan=lifespan)


class AnswerRequest(BaseModel):
    question: str
    context: str


class CorpusRequest(BaseModel):
    question: str
    top_k: int = 5


class AnswerResponse(BaseModel):
    answer: str
    score: float
    confident: bool
    context_used: Optional[str] = None
    sources: list[str] = []
    latency: dict


class IngestResponse(BaseModel):
    added: int
    total: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "corpus_size": len(corpus),
    }


@app.get("/metrics")
def metrics():
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    return engine.latency_stats()


@app.get("/corpus/stats")
def corpus_stats():
    topics: dict[str, int] = {}
    for p in corpus:
        key = p.get("topic") or p.get("source") or "unknown"
        topics[key] = topics.get(key, 0) + 1
    return {"total_passages": len(corpus), "by_topic": topics}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    result = engine.answer(req.question, req.context)
    confident = result["score"] >= SCORE_THRESHOLD
    return AnswerResponse(
        answer=result["answer"] if confident else "Not enough information in the provided context.",
        score=result["score"],
        confident=confident,
        latency=result["latency"],
    )


@app.post("/answer/corpus", response_model=AnswerResponse)
def answer_from_corpus(req: CorpusRequest):
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    if not corpus:
        raise HTTPException(503, "Corpus is empty -- ingest some documents first")

    passages = retriever.retrieve(req.question, top_k=req.top_k)
    context = " ".join(p["text"] for p in passages)
    sources = list({p.get("source") or p.get("topic") or "unknown" for p in passages})

    result = engine.answer(req.question, context)
    confident = result["score"] >= SCORE_THRESHOLD

    return AnswerResponse(
        answer=result["answer"] if confident else "No confident answer found in the corpus.",
        score=result["score"],
        confident=confident,
        context_used=context[:500] + "..." if len(context) > 500 else context,
        sources=sources,
        latency=result["latency"],
    )


@app.post("/corpus/ingest", response_model=IngestResponse)
async def ingest_text(text: str = Form(...), source: str = Form("upload")):
    """
    Add text to the running corpus without restarting the server.
    Split into sections with blank lines -- each becomes a passage.
    """
    global corpus, retriever

    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 80]
    if not chunks:
        raise HTTPException(
            400,
            "No usable chunks. Separate paragraphs with blank lines; each needs >80 chars."
        )

    start_id = max((p["id"] for p in corpus), default=-1) + 1
    new_passages = [
        {"id": start_id + i, "source": source, "topic": source, "text": chunk}
        for i, chunk in enumerate(chunks)
    ]

    corpus.extend(new_passages)
    retriever.rebuild(corpus)

    return IngestResponse(added=len(new_passages), total=len(corpus))
