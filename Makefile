PYTHON := python
PIP    := pip

.PHONY: install demo-corpus train export quantize benchmark serve ui smoke test all clean

# CPU by default; for CUDA: make install TORCH_INDEX=https://download.pytorch.org/whl/cu128
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu
install:
	$(PIP) install torch --index-url $(TORCH_INDEX)
	$(PIP) install -r requirements.txt

# Build the demo corpus (XR/remote-assist domain)
demo-corpus:
	$(PYTHON) data/build_demo_corpus.py

# Fine-tune on whatever corpus.json + qa_dataset.json are in data/
train:
	$(PYTHON) train/train.py

# Export fine-tuned PyTorch model to ONNX FP32
export:
	$(PYTHON) quantize/export_onnx.py

# INT8 quantize + run latency benchmark (FP32 CPU vs INT8 CPU, GPU if available)
quantize:
	$(PYTHON) quantize/quantize_int8.py

# Ingest your own documents into corpus.json
# Usage: make ingest SOURCE=path/to/docs/
ingest:
	$(PYTHON) ingest.py --source $(SOURCE) --output data/corpus.json

serve:
	uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload

ui:
	$(PYTHON) gradio_app.py

# Smoke test — model-free (BM25 only), writes only to a temp dir
smoke:
	$(PYTHON) -m tests.smoke_test

test:
	$(PYTHON) -m pytest -q

# Full pipeline from scratch
all: demo-corpus train export quantize

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
