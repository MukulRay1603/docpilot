"""
Export the fine-tuned QA model to ONNX.

Usage:
    python quantize/export_onnx.py

Input:  models/qa_finetuned/
Output: models/qa_onnx/model.onnx
"""

import sys
# Windows cp1252 console fix
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_DIR = Path(__file__).parent.parent / "models" / "qa_finetuned"
ONNX_DIR = Path(__file__).parent.parent / "models" / "qa_onnx"
ONNX_PATH = ONNX_DIR / "model.onnx"
MAX_LENGTH = 384


def export():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)
    model.eval()

    # Save tokenizer alongside ONNX for the inference server
    tokenizer.save_pretrained(ONNX_DIR)

    # Dummy input for tracing
    dummy_text = "What is waveguide display?"
    dummy_ctx = "Waveguide displays use diffractive optical elements to couple light."
    enc = tokenizer(dummy_text, dummy_ctx, max_length=MAX_LENGTH,
                    truncation=True, padding="max_length", return_tensors="pt")

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    inputs = (input_ids, attention_mask)
    input_names = ["input_ids", "attention_mask"]

    dynamic_axes = {
        "input_ids":      {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "start_logits":   {0: "batch"},
        "end_logits":     {0: "batch"},
    }

    print(f"Exporting to ONNX -> {ONNX_PATH}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(ONNX_PATH),
            input_names=input_names,
            output_names=["start_logits", "end_logits"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,   # use legacy TorchScript exporter; stable on Windows
        )

    import onnx
    onnx.checker.check_model(str(ONNX_PATH))
    size_mb = ONNX_PATH.stat().st_size / 1e6
    print(f"ONNX export OK - {size_mb:.1f} MB")


if __name__ == "__main__":
    export()
