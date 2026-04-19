import os
from pathlib import Path

ROOT = Path(__file__).parent

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models" / "xr_qa_int8"))
CORPUS_PATH = Path(os.getenv("CORPUS_PATH", ROOT / "data" / "xr_corpus.json"))
