"""Dataset loading for train.py, kept free of torch/transformers imports so it
can be unit-tested without the ML stack or network access.

Accepts two layouts:
  1. {"train": {"data": [...]}, "validation": {"data": [...]}}  (explicit splits)
  2. {"version": ..., "data": [...]}  -- what data/build_demo_corpus.py writes.
     Split deterministically: a QA pair goes to validation when
     sha256(id) mod 100 < val_pct, so the split is stable across runs and
     machines and doesn't depend on Python's hash seed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class DatasetFormatError(ValueError):
    pass


def _flatten(articles: list) -> list[dict]:
    rows = []
    for article in articles:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                answers = qa["answers"]
                # SQuAD JSON stores a list of {"text", "answer_start"} dicts;
                # HF preprocessing wants {"text": [...], "answer_start": [...]}.
                if isinstance(answers, list):
                    answers = {
                        "text": [a["text"] for a in answers],
                        "answer_start": [a["answer_start"] for a in answers],
                    }
                if not answers["text"]:
                    continue  # unanswerable items are not used for training
                rows.append({
                    "id": qa["id"],
                    "context": ctx,
                    "question": qa["question"],
                    "answers": answers,
                })
    return rows


def is_validation(qa_id: str, val_pct: int = 15) -> bool:
    bucket = int(hashlib.sha256(qa_id.encode("utf-8")).hexdigest(), 16) % 100
    return bucket < val_pct


def load_splits(path: Path, val_pct: int = 15) -> tuple[list[dict], list[dict]]:
    """Return (train_rows, val_rows). Raises DatasetFormatError on bad input."""
    path = Path(path)
    if not path.exists():
        raise DatasetFormatError(
            f"{path} not found. Run `make demo-corpus` first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))

    if "train" in raw and "validation" in raw:
        train = _flatten(raw["train"]["data"])
        val = _flatten(raw["validation"]["data"])
    elif "data" in raw:
        rows = _flatten(raw["data"])
        train = [r for r in rows if not is_validation(r["id"], val_pct)]
        val = [r for r in rows if is_validation(r["id"], val_pct)]
    else:
        raise DatasetFormatError(
            f"{path}: expected a 'data' key or 'train'/'validation' keys, "
            f"got {sorted(raw)}"
        )

    if not train or not val:
        raise DatasetFormatError(
            f"{path}: need at least one train and one validation example "
            f"(got {len(train)} / {len(val)})"
        )
    return train, val
