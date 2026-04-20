"""
INT8 post-training quantization of the ONNX model using ONNX Runtime.

Usage:
    python quantize/quantize_int8.py

Input:  models/qa_onnx/model.onnx
Output: models/qa_int8/model_int8.onnx

Then runs a latency benchmark comparing FP32 vs INT8.
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
from pathlib import Path
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

ONNX_DIR = Path(__file__).parent.parent / "models" / "qa_onnx"
INT8_DIR = Path(__file__).parent.parent / "models" / "qa_int8"
FP32_PATH = ONNX_DIR / "model.onnx"
INT8_PATH = INT8_DIR / "model_int8.onnx"
MAX_LENGTH = 384
BENCH_ITERS = 50


def quantize():
    INT8_DIR.mkdir(parents=True, exist_ok=True)
    print("Quantizing to INT8 (dynamic) ...")
    quantize_dynamic(
        model_input=str(FP32_PATH),
        model_output=str(INT8_PATH),
        weight_type=QuantType.QInt8,
    )
    fp32_mb = FP32_PATH.stat().st_size / 1e6
    int8_mb = INT8_PATH.stat().st_size / 1e6
    print(f"FP32: {fp32_mb:.1f} MB  ->  INT8: {int8_mb:.1f} MB  "
          f"(compression {fp32_mb / int8_mb:.1f}x)")


def _make_dummy_inputs(seq_len: int = MAX_LENGTH) -> dict:
    rng = np.random.default_rng(0)
    return {
        "input_ids": rng.integers(1000, 30000, (1, seq_len)).astype(np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
    }


def benchmark():
    dummy = _make_dummy_inputs()
    results = {}

    # ── GPU benchmark (FP32 only – INT8 dynamic quant adds dequant overhead on GPU)
    if "CUDAExecutionProvider" in ort.get_available_providers():
        print("\n--- GPU (CUDA) ---")
        sess = ort.InferenceSession(str(FP32_PATH),
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        for _ in range(5):
            sess.run(None, dummy)
        latencies = []
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter()
            sess.run(None, dummy)
            latencies.append((time.perf_counter() - t0) * 1000)
        latencies.sort()
        p95_gpu = latencies[int(len(latencies) * 0.95)]
        print(f"FP32 GPU  P50={latencies[len(latencies)//2]:.1f} ms  P95={p95_gpu:.1f} ms")
        results["FP32_GPU"] = p95_gpu

    # ── CPU benchmark: FP32 vs INT8
    # INT8 dynamic quantization shines on CPU (XR headset / edge device target)
    print("\n--- CPU (target: XR headset / constrained hardware) ---")
    for label, path in [("FP32 CPU", FP32_PATH), ("INT8 CPU", INT8_PATH)]:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for _ in range(5):
            sess.run(None, dummy)
        latencies = []
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter()
            sess.run(None, dummy)
            latencies.append((time.perf_counter() - t0) * 1000)
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        results[label] = p95
        print(f"{label:8s}  P50={p50:.1f} ms  P95={p95:.1f} ms")

    speedup = results["FP32 CPU"] / results["INT8 CPU"]
    print(f"\nINT8 CPU P95 speedup vs FP32 CPU: {speedup:.1f}x  "
          f"({results['FP32 CPU']:.0f} ms -> {results['INT8 CPU']:.0f} ms)")
    print("(INT8 dynamic quantization targets CPU/NPU inference on XR headsets)")


if __name__ == "__main__":
    quantize()
    print()
    benchmark()
