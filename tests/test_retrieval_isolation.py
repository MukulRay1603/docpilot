"""Fix 2 (smoke isolation) and fix 3 (BM25 fallback when models can't load)."""
import os
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PASSAGES = [
    {"id": 0, "text": "Waveguide displays use diffractive optics to project light."},
    {"id": 1, "text": "Inside-out SLAM tracking uses onboard cameras."},
    {"id": 2, "text": "INT8 quantization reduces model size."},
]


def _real_files():
    return {p: (p.stat().st_mtime if p.exists() else None)
            for p in (ROOT / "data" / "audit.db", ROOT / "data" / "chroma_db")}


def test_smoke_does_not_touch_real_data_and_is_model_free():
    before = _real_files()
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    env.pop("CHROMA_DIR", None)
    env.pop("AUDIT_DB", None)
    r = subprocess.run([sys.executable, "-m", "tests.smoke_test"], cwd=ROOT,
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _real_files() == before


def test_chroma_dir_read_at_construction(monkeypatch, tmp_path):
    from serve import retrieval
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "x"))
    r = retrieval.Retriever(PASSAGES, use_dense=False, use_reranker=False)
    assert r.persist_dir == str(tmp_path / "x")


def test_falls_back_to_bm25_when_embedding_model_fails(monkeypatch, tmp_path):
    from serve import retrieval

    def boom(*a, **k):
        raise OSError("We couldn't connect to 'https://huggingface.co'")

    monkeypatch.setattr(retrieval, "_HAS_DENSE", True)
    monkeypatch.setattr(retrieval, "_HAS_RERANKER", True)
    monkeypatch.setattr(retrieval, "SentenceTransformer", boom, raising=False)
    monkeypatch.setattr(retrieval, "CrossEncoder", boom, raising=False)
    monkeypatch.setattr(retrieval, "chromadb", types.SimpleNamespace(), raising=False)
    monkeypatch.delenv("DOCPILOT_BM25_ONLY", raising=False)

    r = retrieval.Retriever(PASSAGES, persist_dir=str(tmp_path / "c"))
    assert r.mode == "bm25"
    assert "huggingface" in r.dense_error
    assert r.retrieve("waveguide optics", top_k=1)[0]["id"] == 0
    # A failed model load must not leave an index tag behind
    assert not (tmp_path / "c" / "embed_model.txt").exists()
