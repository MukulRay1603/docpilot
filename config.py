import os
from pathlib import Path

ROOT = Path(__file__).parent

MODEL_DIR   = Path(os.getenv("MODEL_DIR",   ROOT / "models" / "qa_int8"))
CORPUS_PATH = Path(os.getenv("CORPUS_PATH", ROOT / "data"   / "corpus.json"))

# Confidence threshold: below this the model is probably guessing a span.
# -3.0 works well empirically. Set SCORE_THRESHOLD env var to tune it.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "-3.0"))
