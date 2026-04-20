import os
from pathlib import Path

ROOT = Path(__file__).parent

MODEL_DIR   = Path(os.getenv("MODEL_DIR",   ROOT / "models" / "qa_int8"))
CORPUS_PATH = Path(os.getenv("CORPUS_PATH", ROOT / "data"   / "corpus.json"))

# Confidence threshold for the extractive span. -3.0 works well empirically.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "-3.0"))

# ChromaDB persist path for the vector index (survives restarts)
CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT / "data" / "chroma_db"))
