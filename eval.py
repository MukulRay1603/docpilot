"""
Evaluate the ONNX INT8 QA model (+ retrieval) on the demo corpus.

Usage:
    python eval.py                      # uses INT8 model + retrieval
    python eval.py --no-retrieval       # feed gold context directly (upper bound)
    python eval.py --top-k 3            # number of passages to retrieve

Outputs F1 and EM across all 96 QA pairs from the demo corpus.
"""

import argparse
import json
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR, CORPUS_PATH, CHROMA_DIR
from serve.inference import QAEngine
from serve.retrieval import Retriever


# ---------------------------------------------------------------------------
# SQuAD-style F1 / EM (token-level)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    p_toks = _normalise(pred).split()
    g_toks = _normalise(gold).split()
    if not p_toks or not g_toks:
        return 1.0 if p_toks == g_toks else 0.0
    common = sum(
        min(p_toks.count(t), g_toks.count(t)) for t in set(p_toks) & set(g_toks)
    )
    if common == 0:
        return 0.0
    precision = common / len(p_toks)
    recall    = common / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return _normalise(pred) == _normalise(gold)


# ---------------------------------------------------------------------------
# Load dataset (flat list of {question, context, gold_answers})
# ---------------------------------------------------------------------------

def load_examples(dataset_path: Path) -> list[dict]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    examples = []
    # support both flat {"data": [...]} and split {"train": ..., "validation": ...}
    if "data" in raw:
        entries = raw["data"]
    elif "validation" in raw:
        entries = raw["validation"]["data"]
    else:
        raise ValueError("Unrecognised QA dataset format")

    for article in entries:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible"):
                    continue
                answers = [a["text"] for a in qa["answers"] if a["text"].strip()]
                if not answers:
                    continue
                examples.append({
                    "id":       qa["id"],
                    "question": qa["question"],
                    "context":  ctx,
                    "answers":  answers,
                })
    return examples


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-retrieval", action="store_true",
                        help="Feed gold context directly (measures model ceiling)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of passages to retrieve (default 5)")
    parser.add_argument("--dataset", default=str(ROOT / "data" / "qa_dataset.json"))
    parser.add_argument("--score-threshold", type=float, default=-3.0,
                        help="Min QA model confidence to emit an answer")
    args = parser.parse_args()

    print("Loading QA engine...")
    engine = QAEngine(MODEL_DIR)

    retriever = None
    if not args.no_retrieval:
        print("Loading corpus + retriever...")
        corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        retriever = Retriever(corpus, persist_dir=CHROMA_DIR)

    print(f"Loading eval examples from {args.dataset}...")
    examples = load_examples(Path(args.dataset))
    print(f"  {len(examples)} examples loaded")
    print()

    f1_scores, em_scores, skipped = [], [], 0

    for i, ex in enumerate(examples, 1):
        if args.no_retrieval:
            context = ex["context"]
        else:
            passages = retriever.retrieve(ex["question"], top_k=args.top_k)
            if not passages:
                skipped += 1
                continue
            context = " ".join(p["text"] for p in passages)

        result = engine.answer(ex["question"], context)
        pred   = result["answer"]
        score  = result["score"]

        if score < args.score_threshold or not pred.strip():
            pred = ""

        best_f1 = max(token_f1(pred, g) for g in ex["answers"])
        best_em = max(1.0 if exact_match(pred, g) else 0.0 for g in ex["answers"])

        f1_scores.append(best_f1)
        em_scores.append(best_em)

        if i % 20 == 0 or i == len(examples):
            print(f"  [{i:3d}/{len(examples)}]  running F1={100*sum(f1_scores)/len(f1_scores):.1f}  EM={100*sum(em_scores)/len(em_scores):.1f}")

    n = len(f1_scores)
    print()
    print("=" * 50)
    print(f"  Examples evaluated : {n}  (skipped: {skipped})")
    print(f"  F1                 : {100 * sum(f1_scores) / n:.2f}%")
    print(f"  Exact Match (EM)   : {100 * sum(em_scores) / n:.2f}%")
    mode = "gold context (no retrieval)" if args.no_retrieval else f"hybrid retrieval top-{args.top_k}"
    print(f"  Mode               : {mode}")
    print("=" * 50)

    lat = engine.latency_stats()
    if lat:
        print(f"\n  Latency (n={lat['n']}):  P50={lat['p50_ms']}ms  P95={lat['p95_ms']}ms")


if __name__ == "__main__":
    main()
