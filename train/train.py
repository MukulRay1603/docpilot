"""
Fine-tune a transformer for extractive QA on the XR corpus.

Works on:
  - Google Colab Pro  (change DEVICE to 'cuda' automatically)
  - Local RTX 3060    (CUDA detected automatically)
  - CPU fallback

Usage:
    pip install -r train/requirements.txt
    python data/generate_corpus.py          # build data first
    python train/train.py

Outputs:
    models/xr_qa_finetuned/   – HuggingFace model dir (PyTorch weights)
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
import numpy as np
import collections

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = "deepset/roberta-base-squad2"   # already SQuAD-tuned; great starting point
DATA_PATH = Path(__file__).parent.parent / "data" / "xr_qa_squad.json"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "xr_qa_finetuned"

MAX_LENGTH = 384
DOC_STRIDE = 128
BATCH_SIZE = 4               # RTX 3060 laptop 6 GB – effective batch = 4×4 = 16
GRAD_ACCUM_STEPS = 4
EPOCHS = 4
LR = 2e-5
WARMUP_RATIO = 0.1
N_BEST = 20
MAX_ANSWER_LENGTH = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'N/A'}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_squad_json(path: Path) -> DatasetDict:
    raw = json.loads(path.read_text())

    def flatten(split_data):
        rows = []
        for article in split_data["data"]:
            for para in article["paragraphs"]:
                ctx = para["context"]
                for qa in para["qas"]:
                    rows.append({
                        "id": qa["id"],
                        "context": ctx,
                        "question": qa["question"],
                        "answers": qa["answers"],
                    })
        return rows

    return DatasetDict({
        "train": Dataset.from_list(flatten(raw["train"])),
        "validation": Dataset.from_list(flatten(raw["validation"])),
    })


# ---------------------------------------------------------------------------
# Tokenise
# ---------------------------------------------------------------------------
def preprocess_training(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]

    start_positions, end_positions = [], []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        seq_ids = inputs.sequence_ids(i)
        ctx_start = next(j for j, s in enumerate(seq_ids) if s == 1)
        ctx_end = len(seq_ids) - 1 - next(j for j, s in enumerate(reversed(seq_ids)) if s == 1)

        if offset[ctx_start][0] > end_char or offset[ctx_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            sp = ctx_start
            while sp <= ctx_end and offset[sp][0] <= start_char:
                sp += 1
            start_positions.append(sp - 1)

            ep = ctx_end
            while ep >= ctx_start and offset[ep][1] >= end_char:
                ep -= 1
            end_positions.append(ep + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    for i in range(len(inputs["input_ids"])):
        example_ids.append(examples["id"][sample_map[i]])
        seq_ids = inputs.sequence_ids(i)
        inputs["offset_mapping"][i] = [
            (o if seq_ids[k] == 1 else None)
            for k, o in enumerate(inputs["offset_mapping"][i])
        ]
    inputs["example_id"] = example_ids
    return inputs


# ---------------------------------------------------------------------------
# Metric (F1 + EM)
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    import re, string
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def compute_f1_em(prediction: str, ground_truth: str):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    em = float(normalize_answer(prediction) == normalize_answer(ground_truth))
    return f1, em


def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer):
    all_start_logits, all_end_logits = raw_predictions
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predictions = {}
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        best_answer = None
        best_score = float("-inf")

        for feature_index in example_to_features[example_id]:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indices = np.argsort(start_logits)[-N_BEST:][::-1]
            end_indices = np.argsort(end_logits)[-N_BEST:][::-1]

            for start in start_indices:
                for end in end_indices:
                    if (offsets[start] is None or offsets[end] is None
                            or end < start
                            or end - start + 1 > MAX_ANSWER_LENGTH):
                        continue
                    score = start_logits[start] + end_logits[end]
                    if score > best_score:
                        best_score = score
                        best_answer = context[offsets[start][0]: offsets[end][1]]

        predictions[example_id] = best_answer or ""
    return predictions


def make_compute_metrics(eval_dataset, validation_features, tokenizer):
    def compute_metrics(p: EvalPrediction):
        preds = postprocess_qa_predictions(
            eval_dataset, validation_features, p.predictions, tokenizer
        )
        f1_scores, em_scores = [], []
        for example in eval_dataset:
            pred = preds[example["id"]]
            gt = example["answers"]["text"][0]
            f1, em = compute_f1_em(pred, gt)
            f1_scores.append(f1)
            em_scores.append(em)
        return {
            "f1": round(np.mean(f1_scores) * 100, 2),
            "exact_match": round(np.mean(em_scores) * 100, 2),
        }
    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(BASE_MODEL)

    raw_datasets = load_squad_json(DATA_PATH)
    print(f"Loaded {len(raw_datasets['train'])} train / {len(raw_datasets['validation'])} val examples")

    train_dataset = raw_datasets["train"].map(
        lambda ex: preprocess_training(ex, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    validation_features = raw_datasets["validation"].map(
        lambda ex: preprocess_validation(ex, tokenizer),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )
    eval_dataset = validation_features.remove_columns(["example_id", "offset_mapping"])

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="no",          # skip per-epoch eval; we compute manually after
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        fp16=(DEVICE == "cuda"),
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DefaultDataCollator(),
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nModel saved to {OUTPUT_DIR}")

    # Manual evaluation with span-extraction postprocessing
    print("\nRunning post-training evaluation ...")
    raw_preds = trainer.predict(eval_dataset)
    preds = postprocess_qa_predictions(
        raw_datasets["validation"], validation_features, raw_preds.predictions, tokenizer
    )
    f1_scores, em_scores = [], []
    for example in raw_datasets["validation"]:
        pred = preds[example["id"]]
        gt = example["answers"]["text"][0]
        f1, em = compute_f1_em(pred, gt)
        f1_scores.append(f1)
        em_scores.append(em)
    import numpy as np
    print(f"\nFinal  F1: {np.mean(f1_scores)*100:.1f}%   EM: {np.mean(em_scores)*100:.1f}%")


if __name__ == "__main__":
    main()
