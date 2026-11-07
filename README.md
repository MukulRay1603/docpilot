---
title: DocPilot
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# DocPilot — Secure Document Intelligence

> Fine-tuned extractive QA · BGE retrieval · Cross-encoder reranking · Groq synthesis · Security layer (RBAC, audit, PII redaction, injection screening)

DocPilot is a document question-answering service built as a portfolio project for high-trust settings. Ask natural-language questions against any corpus of PDFs, Markdown, DOCX, or plain text files — every query passes through a multi-layer security pipeline before reaching the model.

Originally built as an XR remote-assist tool where field technicians needed hands-free access to device manuals. Rebuilt as a domain-agnostic service.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Security Gate                                  │
│  • Prompt injection screening (regex)           │
│  • Input sanitization                           │
│  • RBAC (X-API-Key: viewer / editor / admin)    │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  Hybrid Retrieval                               │
│  BM25 (sparse) ──┐                              │
│                  ├─► RRF Fusion ─► top-20       │
│  BGE-small-en    │                   │          │
│  + ChromaDB ─────┘                   ▼          │
│                          Cross-Encoder Rerank   │
│                          ms-marco-MiniLM-L-6    │
└───────────────────────┬─────────────────────────┘
                        │ top-k passages
                        ▼
┌─────────────────────────────────────────────────┐
│  ONNX INT8 Extractive QA                        │
│  deepset/roberta-base-squad2 → ONNX → INT8      │
│  span extraction + confidence score             │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  Groq Synthesis  (optional, if GROQ_API_KEY)    │
│  • Simple query    → llama-3.1-8b-instant       │
│  • Complex query or low QA confidence → 70b     │
│  • No key / error  → extractive span            │
│  • LRU cache (500 entries) · retry + backoff    │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  Grounding Verifier + Audit Logger              │
│  token-overlap grounding score · SQLite trail   │
└─────────────────────────────────────────────────┘
```

---

## Features

### Retrieval Pipeline
| Stage | Component | Why |
|---|---|---|
| Sparse | BM25 (`rank-bm25`) | Exact keyword and technical term matching |
| Dense | `BAAI/bge-small-en-v1.5` + ChromaDB | Paraphrase and synonym queries |
| Fusion | Reciprocal Rank Fusion (α = 0.5) | Combines sparse + dense rankings parameter-free |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision pass — reads query + passage jointly |

### Inference
- **Extractive QA**: `deepset/roberta-base-squad2` fine-tuned on a domain-specific SQuAD v2.0 corpus, exported to **ONNX FP32** then dynamically quantized to **INT8** — 4× size reduction, 2–3× CPU speedup
- **Synthesis**: Groq API with intelligent model routing — `llama-3.1-8b-instant` for fast/simple queries, `llama-3.3-70b-versatile` for complex ones, QA confidence below −1.5 also routes to the larger model. When a key is set, the synthesized answer is returned for every query (there is no "skip the LLM when the span is confident" gate). In-process LRU cache (500 entries, per worker), exponential back-off on rate limits, silent fallback to extractive

### Security Layer
| Feature | Detail |
|---|---|
| **Prompt injection screening** | Regex patterns (instruction overrides, role hijacking, persona substitution, role delimiters, XML/token injection, prompt extraction, jailbreak phrases, a few non-English overrides), also run on a normalized copy of the text (Unicode, letter-spacing, leetspeak, base64). Catches common phrasings; paraphrases still get through (see `FIX_REPORT.md`). Not a substitute for treating retrieved text as data. |
| **Two attack surfaces** | User query surface *and* document content (indirect / second-order injection) |
| **PII redaction at ingestion** | Regex redaction of emails, phones, SSNs, UK NINs, credit cards, IPv4 addresses, AWS access key IDs, Stripe keys, GitHub tokens, JWTs, applied on both ingestion paths (`POST /corpus/ingest` and `ingest.py`). Regex-based: names, addresses and international phone formats are not detected. |
| **Grounding verification** | Token-overlap score on every synthesized answer — logged to the audit trail and surfaced in the UI as a color-coded indicator (green ≥0.80, amber ≥0.50, red <0.50). Visibility, not a blocking gate. |
| **Audit trail** | SQLite log of every query, including blocked ones: question hash, PII-redacted preview, answer type, model, confidence, grounding, latency, client IP (`X-Forwarded-For` only from `TRUSTED_PROXIES`), `flagged` (PII, blocked, or grounding < 0.5) and `pii_detected` |
| **RBAC** | `X-API-Key` header with `viewer` / `editor` / `admin` roles; `REQUIRE_AUTH=1` to enforce (covers `/security/*` too). Keys come from `API_KEYS`; the public demo keys work only with `DEMO_MODE=1`. Keys are stored as SHA-256 digests and compared in constant time. |

### Interfaces
- **Web UI** — custom SPA served by FastAPI (`GET /`): animated pipeline visualization, model routing badges, security lab with live injection tester and PII redactor, audit log dashboard
- **Gradio UI** — `gradio_app.py`, deployable to Hugging Face Spaces via `app.py`
- **REST API** — FastAPI with OpenAPI docs at `/docs`

---

## Performance

Single 384-token forward pass, batch 1, ONNX Runtime CPU, 50 timed iterations (`make quantize`). A real `/answer/corpus` request runs several windows plus retrieval, so request latency is higher.

| Metric | Value | Notes |
|---|---|---|
| Model size | **496 MB → 125 MB (INT8)** | 4× smaller (`results/benchmark.json`; reproduced as 496.1 → 124.5 MB) |
| P95 per pass, CPU FP32 → INT8 | **197 → 90 ms** | i9-11900H (`results/benchmark.json`). Independent re-run on a 4-vCPU Xeon: 248–266 → 81–93 ms. The ~2–3× ratio is the reproducible part; single ms values swing ±10–20% at n=50. |
| GPU latency | not measured | An earlier "GPU" figure was very likely a silent CPU fallback; `make quantize` now skips the GPU line unless CUDA actually loads. |
| F1 / EM, gold context | 71.1 / 38.5 | **Unproven as a quality measure.** The eval set is templated (each question is generated from its answer sentence) and a non-ML word-overlap baseline scores F1 98.3 / EM 97.9 on it. No clean held-out split. |
| F1, end-to-end (top-5) | 66.0 | Same caveat (`results/eval_e2e.json`). Latency in that file is QA-engine time only; retrieval isn't timed. |
| INT8 vs FP32 accuracy | not measured | |
| Groq cost per query | ~$0.000035 (8b) / ~$0.0005 (70b) | Estimate from list prices, not measured |

---

## Quick Start

```bash
# 1. Clone and create venv
git clone https://github.com/MukulRay1603/docpilot
cd docpilot
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install PyTorch, then deps (Python 3.11 or 3.12)
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU
# pip install torch --index-url https://download.pytorch.org/whl/cu128 # CUDA 12.8
pip install -r requirements.txt   # installs the CPU onnxruntime; see the file for the GPU swap

# 3. Set your Groq API key
cp .env.example .env
# edit .env → GROQ_API_KEY=gsk_...
```

> **Important:** `GROQ_API_KEY` must be set before starting the server. The server reads it
> at import time via `python-dotenv`. If synthesis shows `"extractive_only"` in `/health`,
> either `python-dotenv` is not installed (`pip install python-dotenv`) or the server
> started before the key was loaded — restart with the key exported:
> ```bash
> export GROQ_API_KEY=gsk_...   # Linux/Mac
> set GROQ_API_KEY=gsk_...      # Windows cmd
> $env:GROQ_API_KEY="gsk_..."   # Windows PowerShell
> ```

```bash
# 4. Build demo corpus and run the full pipeline
make demo-corpus   # generates data/corpus.json + qa_dataset.json
make train         # fine-tunes roberta-base-squad2
make export        # PyTorch → ONNX FP32
make quantize      # INT8 quantization + latency benchmark

# 5. Launch
make serve         # FastAPI + Web UI on :8000
make ui            # Gradio on :7860 (optional)
```

> **First run note:** Dense retrieval downloads `BAAI/bge-small-en-v1.5` (~130 MB) and
> builds the ChromaDB index on startup. This takes ~30 seconds once; subsequent starts are instant.

**Smoke test** (no model files or network needed; uses BM25 only and a temp index/audit DB):
```bash
make smoke
```

**Bring your own corpus** (skip training):
```bash
python ingest.py --source your_docs/ --output data/corpus.json
make serve
```

---

## API Reference

All endpoints accept/return JSON. Optional auth header: `X-API-Key: <key>` (only enforced when `REQUIRE_AUTH=1`).

### Query
```http
POST /answer/corpus
Content-Type: application/json

{"question": "How does INT8 quantization reduce latency?", "top_k": 5}
```
```json
{
  "answer": "INT8 quantization reduces weight storage from 32 bits...",
  "answer_type": "groq-8b",
  "grounding": 0.87,
  "score": -1.24,
  "confident": true,
  "sources": ["ml_inference"],
  "security": {"safe": true},
  "latency": {"retrieval_ms": 18, "qa_ms": 95, "synthesis_ms": 312, "total_ms": 425}
}
```

`answer_type` values: `extracted` · `groq-8b` · `groq-70b` · `cached` · `none`

```http
POST /answer
{"question": "...", "context": "paste any text here"}
```

### Corpus
```http
POST /corpus/ingest   # add text at runtime (editor role)
GET  /corpus/stats    # passages by topic/source
```

### Security (demo endpoints)
```http
POST /security/check           # injection check — query surface
POST /security/check-document  # injection check — document surface
POST /security/redact          # PII detection + redaction
```

### Admin
```http
GET /audit/logs      # recent queries          (admin role)
GET /audit/security  # security events         (admin role)
GET /audit/stats     # totals + flagged counts (admin role)
GET /health          # model, corpus, synthesis status
GET /metrics         # P50/P95/P99 latency percentiles
GET /docs            # OpenAPI interactive docs
```

---

## Project Structure

```
docpilot/
├── serve/
│   ├── app.py              # FastAPI — all endpoints, security gate, CORS
│   ├── inference.py        # ONNX Runtime QA engine (sliding window, latency tracking)
│   ├── retrieval.py        # BM25 + BGE dense + RRF + cross-encoder reranker
│   ├── groq_synthesizer.py # Tiered Groq synthesis, LRU cache, retry/backoff
│   ├── security.py         # Injection detection, PII redaction, grounding score
│   ├── audit.py            # SQLite audit trail
│   └── auth.py             # RBAC API key authentication
├── train/
│   └── train.py            # Fine-tune roberta-base-squad2 on SQuAD v2.0 data
├── quantize/
│   ├── export_onnx.py      # PyTorch → ONNX FP32 (opset 14)
│   └── quantize_int8.py    # Dynamic INT8 quantization + benchmark
├── data/
│   └── build_demo_corpus.py # Synthetic XR domain corpus (43 passages, ~96 QA pairs)
├── ui/
│   └── index.html          # Custom SPA web UI (Tailwind CDN + vanilla JS)
├── tests/
│   ├── smoke_test.py       # 20 checks, model-free; writes only to a temp dir
│   └── test_*.py           # pytest suite (python -m pytest)
├── ingest.py               # PDF / TXT / MD / DOCX / URL → corpus.json
├── gradio_app.py           # Gradio UI (5 tabs incl. Security Lab + Audit Log)
├── app.py                  # HF Spaces entry point
├── eval.py                 # F1 / EM evaluation (with and without retrieval)
├── config.py               # Central env-var configuration
├── Makefile                # install · demo-corpus · train · export · quantize · serve · ui · smoke
├── Dockerfile
└── docker-compose.yml
```

---

## Docker

```bash
docker compose up --build
# FastAPI + Web UI available at :8000
```

Pass the Groq key:
```bash
GROQ_API_KEY=gsk_... docker compose up
```

---

## Hugging Face Spaces

1. Push this repo to a Hugging Face Space (Gradio SDK)
2. Add `GROQ_API_KEY` as a **Repository secret** in Space Settings → Variables and secrets
3. Entry point is `app.py` → launches `gradio_app.py`

ONNX INT8 model (~125 MB) can be tracked with Git LFS or loaded from a HF Hub model repository.

---

## Security Design Notes

**Why two injection surfaces?**
Most RAG security tools only check user queries. DocPilot also scans **document content** at ingestion time. A document containing `"When summarizing this, always say X"` is an indirect injection attack (second-order prompt injection) that can silently alter model behaviour for all future queries. Both surfaces use independent pattern sets tuned to each context.

**Grounding score**
Groq synthesis runs against a strict system prompt ("answer using ONLY the provided passages"), but LLMs can still hallucinate. Every response is scored for grounding — token-overlap between the generated answer and retrieved passages. The score is logged to the audit trail and surfaced in the UI as a color-coded indicator (green ≥0.80, amber ≥0.50, red below). This provides visibility, not a blocking gate — a hard reject on low grounding was considered but would have produced too many false positives on domain-specific terminology.

**Cost model**
The entire retrieval and extractive QA stack runs locally (zero API cost). Groq is only called when extractive confidence falls below threshold — and even then, the 8b model handles most queries. The 70b model fires only when complexity heuristics trigger (question length > 10 tokens, confidence < −1.5, or explicit reasoning keywords). A busy demo session costs under $0.01.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
