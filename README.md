---
title: XR QA System
emoji: 🥽
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: gradio_app.py
pinned: false
license: mit
---

# XR Corpus RAG — QA System for Extended Reality

An end-to-end Retrieval-Augmented Generation pipeline specialized for **Extended Reality (XR)** domain knowledge.

## What It Does

Ask natural language questions about XR topics and get extractive answers backed by a fine-tuned RoBERTa model:

- **Display optics** — FOV, waveguides, pancake lenses, foveated rendering
- **Tracking & SLAM** — inside-out tracking, 6DoF, visual-inertial odometry
- **Rendering** — reprojection, foveated rendering, frame pacing
- **Spatial audio** — HRTF, latency budgets, ambisonics
- **Networking** — WebRTC, XR cloud streaming, 5G
- **Comfort & ergonomics** — vergence-accommodation conflict, IPD, simulator sickness
- **On-device ML inference** — TensorRT, ONNX, quantization
- **Content creation** — Unity, Unreal, OpenXR authoring tools
- **Platform SDKs** — Meta XR SDK, ARKit, ARCore, OpenXR
- **Enterprise XR** — training simulations, remote assist, digital twins

## Architecture

```
Synthetic XR Corpus (100 passages)
         ↓
  SQuAD v2.0 QA pairs (150+)
         ↓
  Fine-tune RoBERTa (deepset/roberta-base-squad2)
         ↓
  ONNX FP32 export  →  INT8 quantization (4× smaller, 2× faster)
         ↓
  BM25 Retrieval + ONNX Runtime inference
         ↓
  FastAPI (REST)  +  Gradio (UI)
         ↓
  Docker → AWS ECR → EC2 (blue/green deploy)
```

## Quick Start (Local)

### 1. Install PyTorch (GPU — RTX 3060, CUDA 12.8)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
run_pipeline.bat
```
This runs: corpus generation → fine-tuning → ONNX export → INT8 quantization → serve.

### 4. Launch the Gradio UI
```bash
python gradio_app.py
# With public share link:
python gradio_app.py --share
```

### 5. Or run the FastAPI server
```bash
uvicorn serve.app:app --host 0.0.0.0 --port 8000
# Docs at http://localhost:8000/docs
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/answer` | Answer question given explicit context |
| `POST` | `/answer/corpus` | BM25 retrieval from XR corpus + answer |
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Latency percentiles (p50/p95/p99) |

Example:
```bash
curl -X POST http://localhost:8000/answer/corpus \
  -H "Content-Type: application/json" \
  -d '{"question": "How does inside-out tracking work?", "top_k": 3}'
```

## Model Details

| Property | Value |
|----------|-------|
| Base model | `deepset/roberta-base-squad2` |
| Task | Extractive QA (SQuAD v2.0 format) |
| Training epochs | 4 |
| Batch size (effective) | 16 (4 × 4 gradient accum) |
| ONNX FP32 size | ~474 MB |
| ONNX INT8 size | ~120 MB (4× compression) |
| CPU latency (INT8) | 80–150 ms |

## Deployment

### Docker
```bash
docker compose up --build
```

### AWS EC2 (blue/green)
Configure `.env` from `.env.example`, push to GitHub, then CI/CD handles:
1. Lint + smoke test
2. Docker build + push to ECR
3. Zero-downtime deploy to EC2 via SSM

## Project Structure

```
├── data/               # Corpus generation + generated JSON files
├── train/              # Fine-tuning pipeline
├── quantize/           # ONNX export + INT8 quantization
├── serve/              # FastAPI server + ONNX inference engine
├── deploy/             # EC2 setup + blue/green deploy scripts
├── models/             # Fine-tuned weights (gitignored)
├── gradio_app.py       # Gradio demo (also used for HF Spaces)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/  # CI/CD pipeline
```

## Requirements

- Python 3.11–3.12
- PyTorch 2.x with CUDA 12.8 (for training; CPU-only for inference)
- ONNX Runtime (GPU or CPU)
- See `requirements.txt` for full list
