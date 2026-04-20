FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY serve/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY serve/ serve/
COPY ui/ ui/
COPY config.py config.py
COPY ingest.py ingest.py

# Runtime data directories (corpus + models mounted as volumes or built at startup)
RUN mkdir -p data models/qa_int8 models/qa_onnx

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Corpus and model paths — override via docker run -e or docker-compose.yml
ENV CORPUS_PATH=data/corpus.json
ENV MODEL_DIR=models/qa_int8
ENV CHROMA_DIR=data/chroma_db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
