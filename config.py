import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).parent

MODEL_DIR   = Path(os.getenv("MODEL_DIR",   str(ROOT / "models" / "qa_int8")))
CORPUS_PATH = Path(os.getenv("CORPUS_PATH", str(ROOT / "data"   / "corpus.json")))
CHROMA_DIR  = os.getenv("CHROMA_DIR",  str(ROOT / "data" / "chroma_db"))

# Extractive span confidence threshold; -3.0 works well empirically
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "-3.0"))

# Synthesis: set GROQ_API_KEY to enable Groq; falls back to extractive if unset
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
