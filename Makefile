PYTHON := python
PIP    := pip

.PHONY: install corpus train export quantize serve ui all clean

install:
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	$(PIP) install -r requirements.txt

corpus:
	$(PYTHON) data/generate_corpus.py

train:
	$(PYTHON) train/train.py

export:
	$(PYTHON) quantize/export_onnx.py

quantize:
	$(PYTHON) quantize/quantize_int8.py

serve:
	uvicorn serve.app:app --host 0.0.0.0 --port 8000

ui:
	$(PYTHON) gradio_app.py

all: corpus train export quantize

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
