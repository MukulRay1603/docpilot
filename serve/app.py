import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import CORPUS_PATH, MODEL_DIR, SCORE_THRESHOLD
from serve.audit import (
    get_recent_queries,
    get_security_events,
    log_query,
    log_security_event,
    stats as audit_stats,
)
from serve.auth import get_role, require_role
from serve.groq_synthesizer import (
    cache_stats as groq_cache_stats,
    is_available as groq_up,
    synthesize,
)
from serve.inference import QAEngine
from serve.retrieval import Retriever
from serve.security import (
    MAX_QUESTION_LEN,
    check_document_injection,
    check_injection,
    grounding_score,
    redact_pii,
    sanitize_input,
)

engine:    Optional[QAEngine]   = None
corpus:    list[dict]           = []
retriever: Optional[Retriever]  = None


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
        print(f"[app] Loaded {len(data)} passages from {path.name}")
        return data, Retriever(data)
    print(f"[app] No corpus at {path} — starting empty")
    return [], Retriever([])


app = FastAPI(title="DocPilot", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_UI_DIR = Path(__file__).parent.parent / "ui"
if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR)), name="ui")


@app.get("/", include_in_schema=False)
def root():
    index = _UI_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"name": "DocPilot API", "docs": "/docs", "version": "3.0.0"}


class AnswerRequest(BaseModel):
    question: str
    context:  str


class CorpusRequest(BaseModel):
    question: str
    top_k:    int = 5


class AnswerResponse(BaseModel):
    answer:       str
    score:        float
    confident:    bool
    answer_type:  str = "extracted"
    grounding:    float = 1.0
    context_used: Optional[str] = None
    sources:      list[str] = []
    security:     dict = {}
    latency:      dict = {}


class IngestResponse(BaseModel):
    added:        int
    total:        int
    pii_redacted: bool = False
    pii_types:    list[str] = []


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _guard_question(question: str, ip: str) -> str:
    q = sanitize_input(question, MAX_QUESTION_LEN)
    result = check_injection(q)
    if not result.safe:
        log_security_event(
            event_type=result.threat_type,
            details=f"Query injection: {result.matched!r}",
            ip=ip,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error":   "Prompt injection detected",
                "threat":  result.threat_type,
                "matched": result.matched,
            },
        )
    return q


@app.get("/health")
def health(_role: str = Depends(get_role)):
    return {
        "status":       "ok",
        "model_loaded": engine is not None,
        "corpus_size":  len(corpus),
        "synthesis":    "groq" if groq_up() else "extractive_only",
        "groq_cache":   groq_cache_stats(),
        "audit":        audit_stats(),
    }


@app.get("/metrics")
def metrics(_role: str = Depends(get_role)):
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    return engine.latency_stats()


@app.get("/corpus/stats")
def corpus_stats(_role: str = Depends(get_role)):
    topics: dict[str, int] = {}
    for p in corpus:
        key = p.get("topic") or p.get("source") or "unknown"
        topics[key] = topics.get(key, 0) + 1
    return {"total_passages": len(corpus), "by_topic": topics}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest, request: Request, _role: str = Depends(get_role)):
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    ip = _client_ip(request)
    q  = _guard_question(req.question, ip)

    t0     = time.perf_counter()
    result = engine.answer(q, req.context)
    qa_ms  = round((time.perf_counter() - t0) * 1000, 1)

    confident   = result["score"] >= SCORE_THRESHOLD
    answer_text = result["answer"] if confident else "Not enough information in the provided context."
    g_score     = grounding_score(answer_text, [req.context])

    log_query(
        question=q, answer=answer_text, answer_type="extracted",
        model_used="onnx", confidence=result["score"], grounding=g_score,
        sources=[], latency={"qa_ms": qa_ms, "total_ms": qa_ms}, ip=ip,
    )
    return AnswerResponse(
        answer=answer_text, score=result["score"], confident=confident,
        grounding=g_score, security={"safe": True},
        latency={**result["latency"], "qa_ms": qa_ms, "total_ms": qa_ms},
    )


@app.post("/answer/corpus", response_model=AnswerResponse)
def answer_from_corpus(req: CorpusRequest, request: Request, _role: str = Depends(get_role)):
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    if not corpus:
        raise HTTPException(503, "Corpus is empty — ingest some documents first")

    ip = _client_ip(request)
    q  = _guard_question(req.question, ip)

    t_ret    = time.perf_counter()
    passages = retriever.retrieve(q, top_k=req.top_k)
    ret_ms   = round((time.perf_counter() - t_ret) * 1000, 1)

    context  = " ".join(p["text"] for p in passages)
    sources  = list({p.get("source") or p.get("topic") or "unknown" for p in passages})

    t_qa   = time.perf_counter()
    result = engine.answer(q, context)
    qa_ms  = round((time.perf_counter() - t_qa) * 1000, 1)

    confident     = result["score"] >= SCORE_THRESHOLD
    passage_texts = [p["text"] for p in passages]

    t_synth           = time.perf_counter()
    synth, model_label = synthesize(q, passage_texts, confidence=result["score"])
    synth_ms          = round((time.perf_counter() - t_synth) * 1000, 1)

    if synth:
        final_answer = synth
        answer_type  = model_label
    elif confident and result["answer"]:
        final_answer = result["answer"]
        answer_type  = "extracted"
    else:
        final_answer = "No confident answer found in the corpus."
        answer_type  = "none"

    g_score  = grounding_score(final_answer, passage_texts)
    total_ms = round(ret_ms + qa_ms + synth_ms, 1)

    log_query(
        question=q, answer=final_answer, answer_type=answer_type,
        model_used=model_label or "onnx", confidence=result["score"],
        grounding=g_score, sources=sources,
        latency={"retrieval_ms": ret_ms, "qa_ms": qa_ms,
                 "synthesis_ms": synth_ms, "total_ms": total_ms},
        ip=ip,
    )
    return AnswerResponse(
        answer=final_answer,
        score=result["score"],
        confident=confident,
        answer_type=answer_type,
        grounding=g_score,
        context_used=context[:500] + "…" if len(context) > 500 else context,
        sources=sources,
        security={"safe": True},
        latency={
            "retrieval_ms": ret_ms,
            "qa_ms":        qa_ms,
            "synthesis_ms": synth_ms,
            "total_ms":     total_ms,
        },
    )


@app.post("/corpus/ingest", response_model=IngestResponse)
async def ingest_text(
    text:   str = Form(...),
    source: str = Form("upload"),
    _role:  str = Depends(require_role("editor")),
):
    global corpus, retriever

    pii        = redact_pii(text)
    clean_text = pii.redacted_text

    doc_check = check_document_injection(clean_text)
    if not doc_check.safe:
        log_security_event(
            event_type=f"doc_{doc_check.threat_type}",
            details=f"Indirect injection in upload: {doc_check.matched!r}",
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error":   "Indirect injection pattern detected in document",
                "threat":  doc_check.threat_type,
                "matched": doc_check.matched,
            },
        )

    chunks = [c.strip() for c in clean_text.split("\n\n") if len(c.strip()) > 80]
    if not chunks:
        raise HTTPException(
            400,
            "No usable chunks. Separate paragraphs with blank lines; each needs >80 chars.",
        )

    start_id     = max((p["id"] for p in corpus), default=-1) + 1
    new_passages = [
        {"id": start_id + i, "source": source, "topic": source, "text": chunk}
        for i, chunk in enumerate(chunks)
    ]
    corpus.extend(new_passages)
    retriever.rebuild(corpus)

    return IngestResponse(
        added=len(new_passages),
        total=len(corpus),
        pii_redacted=pii.has_pii,
        pii_types=[f["type"] for f in pii.found],
    )


class InjectionRequest(BaseModel):
    text: str


class InjectionResponse(BaseModel):
    safe:        bool
    threat_type: str = ""
    matched:     str = ""


class PIIRequest(BaseModel):
    text: str


class PIIResponse(BaseModel):
    redacted_text: str
    found:         list[dict]
    has_pii:       bool


@app.post("/security/check", response_model=InjectionResponse)
def security_check(req: InjectionRequest, request: Request):
    q   = sanitize_input(req.text, MAX_QUESTION_LEN)
    res = check_injection(q)
    if not res.safe:
        log_security_event(res.threat_type, f"Demo check: {res.matched!r}", _client_ip(request))
    return InjectionResponse(safe=res.safe, threat_type=res.threat_type, matched=res.matched)


@app.post("/security/check-document", response_model=InjectionResponse)
def security_check_doc(req: InjectionRequest):
    res = check_document_injection(req.text[:10_000])
    return InjectionResponse(safe=res.safe, threat_type=res.threat_type, matched=res.matched)


@app.post("/security/redact", response_model=PIIResponse)
def security_redact(req: PIIRequest):
    result = redact_pii(req.text[:50_000])
    return PIIResponse(
        redacted_text=result.redacted_text,
        found=result.found,
        has_pii=result.has_pii,
    )


@app.get("/audit/logs")
def audit_logs(n: int = 20, _role: str = Depends(require_role("admin"))):
    return {"queries": get_recent_queries(n)}


@app.get("/audit/security")
def security_events(n: int = 20, _role: str = Depends(require_role("admin"))):
    return {"events": get_security_events(n)}


@app.get("/audit/stats")
def audit_summary(_role: str = Depends(require_role("admin"))):
    return audit_stats()
