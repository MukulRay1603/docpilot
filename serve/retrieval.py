"""
Hybrid retrieval: BM25 + dense vectors (ChromaDB) + cross-encoder reranking.

Pipeline
────────
1. BM25   (sparse, exact-keyword)          — rank_bm25
2. Dense  (semantic, paraphrase-aware)     — BAAI/bge-small-en-v1.5 + ChromaDB
3. RRF    (Reciprocal Rank Fusion)         — fuse BM25 + dense rankings
4. Rerank (cross-encoder precision pass)   — cross-encoder/ms-marco-MiniLM-L-6-v2

Steps 1-3 produce N candidates (default 20).
Step 4 reranks those candidates and returns the final top-k.

The cross-encoder reads query and passage *jointly*, giving significantly better
ranking accuracy than bi-encoder cosine similarity alone (+5-8 retrieval points
on BEIR benchmarks) at the cost of ~15 ms CPU per 20 passages — well worth it.

BGE note: bge-small-en-v1.5 benefits from a query instruction prefix on retrieval
queries. Passage embeddings use no prefix. Model compat is tracked via
{CHROMA_DIR}/embed_model.txt; changing the model triggers a clean re-index.
"""

import os
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    _HAS_DENSE = True
except ImportError:
    _HAS_DENSE = False
    print("[retrieval] chromadb / sentence-transformers not installed — BM25 only")

try:
    from sentence_transformers import CrossEncoder
    _HAS_RERANKER = True
except ImportError:
    _HAS_RERANKER = False

CHROMA_DIR    = os.getenv("CHROMA_DIR",   "data/chroma_db")
_EMBED_MODEL  = os.getenv("EMBED_MODEL",  "BAAI/bge-small-en-v1.5")
_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_IS_BGE       = "bge" in _EMBED_MODEL.lower()
_BGE_Q_PREFIX = "Represent this sentence for searching relevant passages: "

RRF_K          = 60     # standard RRF constant
_N_CANDIDATES  = 20     # retrieve this many before reranking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tok(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    return re.sub(r"[^\w\s]", " ", text).split()


def _rrf_fuse(bm25_ranks: list[int], dense_ranks: list[int], n: int, alpha: float = 0.5) -> np.ndarray:
    scores = np.zeros(n)
    for rank, idx in enumerate(bm25_ranks):
        scores[idx] += (1 - alpha) / (RRF_K + rank + 1)
    for rank, idx in enumerate(dense_ranks):
        scores[idx] += alpha / (RRF_K + rank + 1)
    return scores


def _check_model_compat(persist_dir: str, current_model: str) -> bool:
    """
    Return True if the stored embedding model matches current_model.
    On mismatch, overwrite the tag file and signal the caller to reset.
    """
    tag_file = Path(persist_dir) / "embed_model.txt"
    if tag_file.exists():
        stored = tag_file.read_text(encoding="utf-8").strip()
        if stored != current_model:
            print(f"[retrieval] Embedding model changed ({stored} → {current_model}) — clearing index")
            tag_file.write_text(current_model, encoding="utf-8")
            return False
    else:
        tag_file.parent.mkdir(parents=True, exist_ok=True)
        tag_file.write_text(current_model, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, passages: list[dict], persist_dir: str = CHROMA_DIR):
        self.passages = passages
        self._bm25: BM25Okapi | None = (
            BM25Okapi([_tok(p["text"]) for p in passages]) if passages else None
        )
        self._chroma     = None
        self._collection = None
        self._encoder    = None
        self._reranker   = None

        if _HAS_DENSE and passages:
            self._init_dense(persist_dir)

        if _HAS_RERANKER:
            try:
                self._reranker = CrossEncoder(_RERANK_MODEL, max_length=512)
                print(f"[retrieval] Cross-encoder loaded: {_RERANK_MODEL}")
            except Exception as exc:
                print(f"[retrieval] Cross-encoder unavailable: {exc}")

    # ── dense index ──────────────────────────────────────────────────────────

    def _init_dense(self, persist_dir: str) -> None:
        compat = _check_model_compat(persist_dir, _EMBED_MODEL)
        self._encoder = SentenceTransformer(_EMBED_MODEL)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=persist_dir)
        if not compat:
            # embedding model changed — nuke old collection so dimensions don't mismatch
            try:
                client.delete_collection("docpilot_corpus")
            except Exception:
                pass

        self._chroma = client
        self._collection = client.get_or_create_collection(
            "docpilot_corpus",
            metadata={"hnsw:space": "cosine"},
        )
        self._sync_passages(self.passages)

    def _encode_query(self, query: str) -> list[list[float]]:
        q = _BGE_Q_PREFIX + query if _IS_BGE else query
        return self._encoder.encode(
            [q], normalize_embeddings=_IS_BGE, show_progress_bar=False
        ).tolist()

    def _sync_passages(self, passages: list[dict]) -> None:
        if not self._encoder or self._collection is None:
            return
        existing = set(self._collection.get(include=[])["ids"])
        new = [p for p in passages if str(p["id"]) not in existing]
        if not new:
            return
        print(f"[retrieval] Embedding {len(new)} new passages ({_EMBED_MODEL})…")
        texts = [p["text"] for p in new]
        embeddings = self._encoder.encode(
            texts, normalize_embeddings=_IS_BGE, show_progress_bar=False, batch_size=32
        ).tolist()
        self._collection.upsert(
            ids=[str(p["id"]) for p in new],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"topic": p.get("topic", ""), "source": p.get("source", "")} for p in new],
        )

    # ── retrieval ────────────────────────────────────────────────────────────

    def _retrieve_candidates(self, query: str, n: int) -> list[dict]:
        """BM25 + dense RRF fusion → up to n candidates."""
        bm25_scores = np.array(self._bm25.get_scores(_tok(query)))
        bm25_ranks  = np.argsort(-bm25_scores).tolist()

        if self._collection is not None and self._encoder is not None:
            q_emb   = self._encode_query(query)
            n_dense = min(len(self.passages), max(n, 20))
            results = self._collection.query(
                query_embeddings=q_emb,
                n_results=n_dense,
                include=["distances"],
            )
            dense_ids = [int(i) for i in results["ids"][0]]
            id_to_pos = {p["id"]: i for i, p in enumerate(self.passages)}
            dense_ranks = [id_to_pos[i] for i in dense_ids if i in id_to_pos]
            ranked_set  = set(dense_ranks)
            for i in range(len(self.passages)):
                if i not in ranked_set:
                    dense_ranks.append(i)

            fused = _rrf_fuse(bm25_ranks, dense_ranks, len(self.passages))
        else:
            fused = bm25_scores / (bm25_scores.max() + 1e-9)

        top_idx = np.argsort(-fused)[:n]
        return [
            {**self.passages[i], "retrieval_score": float(fused[i])}
            for i in top_idx
        ]

    def _rerank_passages(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Cross-encoder reranking: jointly scores query + each passage."""
        pairs  = [(query, p["text"]) for p in candidates]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [
            {**p, "rerank_score": round(float(s), 4)}
            for s, p in ranked[:top_k]
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.passages or self._bm25 is None:
            return []

        n_candidates = min(len(self.passages), max(top_k * 4, _N_CANDIDATES))
        candidates   = self._retrieve_candidates(query, n_candidates)

        if self._reranker and len(candidates) > top_k:
            return self._rerank_passages(query, candidates, top_k)

        return candidates[:top_k]

    def rebuild(self, passages: list[dict]) -> None:
        self.passages = passages
        self._bm25    = BM25Okapi([_tok(p["text"]) for p in passages]) if passages else None
        if self._collection is not None:
            self._sync_passages(passages)
