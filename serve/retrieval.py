"""
Hybrid retrieval: BM25 + dense vectors via ChromaDB.

BM25 handles exact-term and technical keyword queries well.
ChromaDB (sentence-transformers all-MiniLM-L6-v2) handles paraphrase
and synonyms that BM25 misses -- e.g. "screen refresh" vs "display frame rate".

We fuse the two rankings using Reciprocal Rank Fusion (RRF) which is simple,
parameter-free, and works better than naive score averaging for most corpora.

ChromaDB is persistent: the vector index survives restarts. Pass the same
persist_dir across runs and it only embeds new passages.
"""

import os
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    _HAS_DENSE = True
except ImportError:
    _HAS_DENSE = False
    print("chromadb / sentence-transformers not installed -- BM25 only")

CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")
_EMBED_MODEL = "all-MiniLM-L6-v2"  # 22 MB, fast on CPU, good enough for passage retrieval
RRF_K = 60  # standard RRF constant; higher = more conservative rank fusion


def _tok(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    return re.sub(r"[^\w\s]", " ", text).split()


def _rrf_fuse(bm25_ranks: list[int], dense_ranks: list[int], n: int, alpha: float = 0.5) -> np.ndarray:
    """Fuse two ranked lists using Reciprocal Rank Fusion."""
    scores = np.zeros(n)
    for rank, idx in enumerate(bm25_ranks):
        scores[idx] += (1 - alpha) / (RRF_K + rank + 1)
    for rank, idx in enumerate(dense_ranks):
        scores[idx] += alpha / (RRF_K + rank + 1)
    return scores


class Retriever:
    def __init__(self, passages: list[dict], persist_dir: str = CHROMA_DIR):
        self.passages = passages
        self._bm25 = BM25Okapi([_tok(p["text"]) for p in passages]) if passages else None
        self._chroma = None
        self._encoder = None

        if _HAS_DENSE and passages:
            self._encoder = SentenceTransformer(_EMBED_MODEL)
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._chroma.get_or_create_collection(
                "docpilot_corpus",
                metadata={"hnsw:space": "cosine"},
            )
            self._sync_passages(passages)

    def _sync_passages(self, passages: list[dict]) -> None:
        """Upsert only passages not already in the collection."""
        if not self._encoder:
            return
        existing = set(self._collection.get(include=[])["ids"])
        new = [p for p in passages if str(p["id"]) not in existing]
        if not new:
            return
        print(f"Embedding {len(new)} new passages...")
        texts = [p["text"] for p in new]
        embeddings = self._encoder.encode(texts, show_progress_bar=False, batch_size=32).tolist()
        self._collection.upsert(
            ids=[str(p["id"]) for p in new],
            documents=[p["text"] for p in new],
            embeddings=embeddings,
            metadatas=[{"topic": p.get("topic", ""), "source": p.get("source", "")} for p in new],
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.passages or self._bm25 is None:
            return []

        bm25_scores = np.array(self._bm25.get_scores(_tok(query)))
        bm25_ranks = np.argsort(-bm25_scores).tolist()

        if self._chroma is not None and self._encoder is not None:
            q_emb = self._encoder.encode([query]).tolist()
            # get top-2*top_k from dense; we'll re-rank and take top_k after fusion
            n_dense = min(len(self.passages), max(top_k * 2, 20))
            dense_results = self._collection.query(
                query_embeddings=q_emb,
                n_results=n_dense,
                include=["distances"],
            )
            dense_ids = [int(i) for i in dense_results["ids"][0]]
            # build full dense ranking: dense results first, then the rest unranked
            id_to_pos = {p["id"]: i for i, p in enumerate(self.passages)}
            dense_ranks = [id_to_pos[i] for i in dense_ids if i in id_to_pos]
            # pad with unranked positions (appear at end)
            ranked_set = set(dense_ranks)
            for i in range(len(self.passages)):
                if i not in ranked_set:
                    dense_ranks.append(i)

            fused = _rrf_fuse(bm25_ranks, dense_ranks, len(self.passages))
        else:
            # BM25 only: normalise and return top-k
            fused = bm25_scores / (bm25_scores.max() + 1e-9)

        top_idx = np.argsort(-fused)[:top_k]
        return [
            {**self.passages[i], "retrieval_score": float(fused[i])}
            for i in top_idx
        ]

    def rebuild(self, passages: list[dict]) -> None:
        """Rebuild BM25 in-memory; sync new passages into ChromaDB."""
        self.passages = passages
        self._bm25 = BM25Okapi([_tok(p["text"]) for p in passages]) if passages else None
        if self._chroma is not None:
            self._sync_passages(passages)
