import time
import collections
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MAX_LENGTH = 384
DOC_STRIDE = 128
N_BEST = 20
MAX_ANSWER_LENGTH = 128

_Record = collections.namedtuple("_Record", ["tokenize_ms", "inference_ms", "decode_ms", "total_ms"])


def _cuda_available() -> bool:
    try:
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _input_names(session: ort.InferenceSession) -> list[str]:
    return [inp.name for inp in session.get_inputs()]


ort.InferenceSession.get_inputs_names_ = lambda self: _input_names(self)


class QAEngine:
    def __init__(self, model_dir: Path):
        int8_path = model_dir / "model_int8.onnx"
        fp32_path = model_dir.parent / "xr_qa_onnx" / "model.onnx"
        model_path = int8_path if int8_path.exists() else fp32_path

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if _cuda_available()
            else ["CPUExecutionProvider"]
        )
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir.parent / "xr_qa_onnx")
        self._window: collections.deque = collections.deque(maxlen=200)
        print(f"{model_path.name} loaded | {self.session.get_providers()}")

    def answer(self, question: str, context: str) -> dict:
        t0 = time.perf_counter()

        enc = self.tokenizer(
            question, context,
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="np",
        )
        t_tok = time.perf_counter()

        best_answer, best_score = "", float("-inf")
        offset_maps = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping", None)

        for i in range(enc["input_ids"].shape[0]):
            feed = {k: enc[k][i: i + 1] for k in self.session.get_inputs_names_()}
            outputs = self.session.run(None, feed)
            start_logits, end_logits = outputs[0][0], outputs[1][0]
            offsets = offset_maps[i]

            for s in np.argsort(start_logits)[-N_BEST:][::-1]:
                for e in np.argsort(end_logits)[-N_BEST:][::-1]:
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > MAX_ANSWER_LENGTH:
                        continue
                    score = float(start_logits[s]) + float(end_logits[e])
                    if score > best_score:
                        best_score = score
                        best_answer = context[offsets[s][0]: offsets[e][1]]

        t_inf = time.perf_counter()
        answer_text = best_answer.strip() or "No answer found."
        t_end = time.perf_counter()

        rec = _Record(
            tokenize_ms=round((t_tok - t0) * 1000, 2),
            inference_ms=round((t_inf - t_tok) * 1000, 2),
            decode_ms=round((t_end - t_inf) * 1000, 2),
            total_ms=round((t_end - t0) * 1000, 2),
        )
        self._window.append(rec)

        return {
            "answer": answer_text,
            "score": round(best_score, 4),
            "latency": rec._asdict(),
        }

    def latency_stats(self) -> dict:
        if not self._window:
            return {}
        totals = sorted(r.total_ms for r in self._window)
        n = len(totals)
        return {
            "n": n,
            "p50_ms": totals[n // 2],
            "p95_ms": totals[int(n * 0.95)],
            "p99_ms": totals[int(n * 0.99)],
            "mean_ms": round(sum(totals) / n, 2),
        }


# backwards-compat alias used by gradio_app and earlier imports
XRQAEngine = QAEngine
