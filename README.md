---
title: DocPilot
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: gradio_app.py
pinned: false
license: mit
---

# DocPilot

Extractive QA over your own document corpus. You bring the documents, DocPilot retrieves relevant passages and extracts the answer span using a fine-tuned RoBERTa model running on ONNX Runtime.

Originally built as a remote support tool for an XR company. Field technicians needed to query device manuals and maintenance procedures hands-free while working. This is a cleaned-up, domain-agnostic version of that system.

## What it does

1. You ingest documents (PDF, TXT, MD, DOCX) into a passage corpus using `ingest.py`
2. At query time, BM25 retrieves the most relevant passages
3. A fine-tuned RoBERTa model extracts the answer span from the retrieved context
4. The answer and source passages are returned via REST API or Gradio UI

The demo ships with a synthetic corpus covering XR hardware topics (display optics, tracking, rendering, spatial audio, networking, ML inference, enterprise deployment). Replace it with your own documents.

## Architecture

```
Your documents (PDF / TXT / MD / DOCX)
        ↓  ingest.py
corpus.json  (passages with source metadata)
        ↓  at query time
BM25 retrieval  →  top-k passages
        ↓
RoBERTa QA model (ONNX INT8)  →  answer span
        ↓
FastAPI  /  Gradio UI
```

## Quick start

### 1. Install PyTorch (GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the demo corpus and run the full training pipeline

```bash
make all       # demo-corpus → train → export → quantize
```

Or step by step:

```bash
python data/build_demo_corpus.py   # writes data/corpus.json + data/qa_dataset.json
python train/train.py              # fine-tunes RoBERTa, writes models/qa_finetuned/
python quantize/export_onnx.py     # ONNX FP32, writes models/qa_onnx/model.onnx
python quantize/quantize_int8.py   # INT8 quantize + latency benchmark
```

### 4. Serve

```bash
make serve     # FastAPI on :8000
make ui        # Gradio on :7860
```

### 5. Use your own corpus

```bash
python ingest.py --source path/to/your/docs/   # folder of PDFs / TXT / MD
python ingest.py --source manual.pdf           # single file
python ingest.py --source https://...          # web page (needs: pip install requests beautifulsoup4)
```

Or use the "Your Corpus" tab in the Gradio UI to add text or upload files without restarting.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/answer` | QA with explicit context string |
| `POST` | `/answer/corpus` | BM25 retrieval from corpus + QA |
| `POST` | `/corpus/ingest` | Add text to the running corpus (form: `text`, `source`) |
| `GET`  | `/corpus/stats` | Passage counts by topic/source |
| `GET`  | `/health` | Liveness check |
| `GET`  | `/metrics` | Inference latency percentiles (p50/p95/p99) |

```bash
curl -X POST http://localhost:8000/answer/corpus \
  -H "Content-Type: application/json" \
  -d '{"question": "How does INT8 quantization reduce latency?", "top_k": 3}'
```

## Model and performance

| | |
|---|---|
| Base model | `deepset/roberta-base-squad2` |
| Max sequence length | 384 tokens |
| Training set | ~96 QA pairs (synthetic, SQuAD v2.0 format) |
| ONNX FP32 | ~479 MB |
| ONNX INT8 | ~120 MB |
| **FP32 CPU P95** | ~185 ms (i9-11900H) |
| **INT8 CPU P95** | ~62 ms (i9-11900H, ~3x speedup) |
| **FP32 GPU P95** | ~18 ms (RTX 3060) |

Run `python quantize/quantize_int8.py` to reproduce latency numbers on your hardware.

The training set is small (synthetic demo data). The base model already does well on extractive QA from SQuAD pre-training; fine-tuning on domain data helps mainly with domain-specific terminology.

## Deployment

### Docker (local)

```bash
docker compose up --build
```

### Production

The `deploy/` folder has scripts for a blue/green zero-downtime deployment to AWS EC2 via ECR. The pattern is:

1. Build and push Docker image to a container registry (ECR, GHCR, etc.)
2. On the EC2 instance, run `deploy/deploy.sh <IMAGE_URI>`
3. The script starts a "green" container, health-checks it, switches Nginx upstream, then stops the old "blue" container

This requires an EC2 instance with Docker and Nginx, an ECR repository, and an IAM role with appropriate permissions. The GitHub Actions workflow in `.github/workflows/ci_cd.yml` has commented-out steps showing how to wire up the push and deploy automatically.

AWS isn't required to run the project; it's just the deployment target we used for the original XR remote support tool.

## Retrieval quality

The default retrieval is BM25 (keyword-based). It works well for direct terminology queries but misses paraphrase and synonym matches.

For better retrieval, set `USE_SEMANTIC=1` and install `sentence-transformers`:

```bash
pip install sentence-transformers
USE_SEMANTIC=1 python gradio_app.py
```

This uses `all-MiniLM-L6-v2` (22 MB) and blends semantic similarity with BM25 scores. Cold-start adds ~2 seconds for embedding the corpus; per-query latency increase is minimal.

## Project structure

```
data/
  build_demo_corpus.py   -- generates demo corpus.json + qa_dataset.json
  corpus.json            -- gitignored, generated
  qa_dataset.json        -- gitignored, generated
train/
  train.py               -- fine-tune RoBERTa on qa_dataset.json
quantize/
  export_onnx.py         -- PyTorch to ONNX FP32
  quantize_int8.py       -- dynamic INT8 + latency benchmark
serve/
  app.py                 -- FastAPI endpoints
  inference.py           -- ONNX Runtime QA engine
  retrieval.py           -- BM25 + optional semantic retrieval
ingest.py                -- document ingestion pipeline (PDF/TXT/MD/DOCX/URL)
gradio_app.py            -- Gradio UI
config.py                -- reads MODEL_DIR, CORPUS_PATH, SCORE_THRESHOLD from env
deploy/
  deploy.sh              -- blue/green EC2 deploy script
  ec2_setup.sh           -- one-time EC2 bootstrap
```

## Known limitations

- Extractive only: answers are spans from a single passage, not synthesised across multiple sources
- BM25 misses semantic matches without `USE_SEMANTIC=1`
- The demo corpus is small (43 synthetic passages). Real deployments need real documents.
- No conversation history, no follow-up questions
