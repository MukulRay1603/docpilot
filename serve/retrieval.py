"""
Hybrid BM25 + optional dense retrieval.

BM25 is the default -- no extra model weights, fast on CPU, good enough for
keyword-heavy technical queries. The obvious weakness is synonyms and
paraphrase queries ("screen refresh" vs "display frame rate" will miss).

If you set USE_SEMANTIC=1 and install sentence-transformers, we also run a
dense pass with all-MiniLM-L6-v2 (22 MB) and linearly blend the two scores.
It's a meaningful improvement, especially for conceptual questions.
"""

import os
import re

import numpy as np
from rank_bm25 import BM25Okapi

_SEMANTIC = os.getenv("USE_SEMANTIC", "0") == "1"

try:
    if _SEMANTIC:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Semantic retrieval enabled (all-MiniLM-L6-v2)")
    else:
        _encoder = None
except ImportError:
    _encoder = None
    print("sentence-transformers not installed, falling back to BM25 only")


def _tok(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    return re.sub(r"[^\w\s]", " ", text).split()


class Retriever:
    def __init__(self, passages: list[dict]):
        self.passages = passages
        self._bm25 = BM25Okapi([_tok(p["text"]) for p in passages]) if passages else None
        self._embeddings = None

        if _encoder is not None and passages:
            self._embeddings = _encoder.encode(
                [p["text"] for p in passages],
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.passages or self._bm25 is None:
            return []

        bm25_raw = np.array(self._bm25.get_scores(_tok(query)), dtype=float)
        bm25_max = bm25_raw.max()
        bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

        if self._embeddings is not None and _encoder is not None:
            q_emb = _encoder.encode([query], convert_to_numpy=True)
            norms = (
                np.linalg.norm(self._embeddings, axis=1, keepdims=True)
                * np.linalg.norm(q_emb)
            )
            sem = (self._embeddings @ q_emb.T / (norms + 1e-9)).flatten()
            sem_norm = (sem + 1.0) / 2.0  # shift from [-1,1] to [0,1]
            # 40/60 blend -- BM25 is more reliable for exact-match tech queries,
            # semantic helps with paraphrases. Tune if you have labeled eval data.
            scores = 0.4 * bm25_norm + 0.6 * sem_norm
        else:
            scores = bm25_norm

        top_idx = scores.argsort()[-top_k:][::-1]
        return [
            {**self.passages[i], "retrieval_score": float(scores[i])}
            for i in top_idx
        ]

    def rebuild(self, passages: list[dict]) -> None:
        self.__init__(passages)
