import argparse
import json
import re
from pathlib import Path

import gradio as gr
from rank_bm25 import BM25Okapi

from config import MODEL_DIR, CORPUS_PATH
from serve.inference import QAEngine

def _tok(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    return re.sub(r"[^\w\s]", " ", text).split()


engine = QAEngine(MODEL_DIR)

corpus: list[dict] = []
bm25: BM25Okapi | None = None

if CORPUS_PATH.exists():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    bm25 = BM25Okapi([_tok(doc["text"]) for doc in corpus])
    print(f"Loaded {len(corpus)} passages from {CORPUS_PATH.name}")


def _retrieve(question: str, top_k: int):
    if bm25 is None:
        return [], [], []
    scores = bm25.get_scores(_tok(question))
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [corpus[i]["text"] for i in idx], [float(scores[i]) for i in idx], idx


TOPICS = sorted({doc.get("topic", "unknown") for doc in corpus}) if corpus else []

EXAMPLES = [
    ["What is foveated rendering?"],
    ["How does inside-out tracking work?"],
    ["What latency is required to avoid simulation sickness in spatial audio?"],
    ["What is the vergence-accommodation conflict?"],
    ["How is IPD mismatch measured?"],
    ["What is visual-inertial odometry?"],
    ["How does WebRTC handle XR streaming?"],
]

CSS = """
:root {
    --bg:      #0a0a14;
    --card:    #12121f;
    --border:  #1e1e35;
    --purple:  #7c3aed;
    --indigo:  #4f46e5;
    --cyan:    #06b6d4;
    --text:    #e2e8f0;
    --muted:   #64748b;
}
body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; }

.header-wrap {
    background: linear-gradient(135deg, #0d0d1f 0%, #160d2e 50%, #0d1a2e 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 4px;
    position: relative;
    overflow: hidden;
}
.header-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 15% 50%, rgba(124,58,237,.12) 0%, transparent 100%),
        radial-gradient(ellipse 50% 60% at 85% 50%, rgba(6,182,212,.08) 0%, transparent 100%);
    pointer-events: none;
}

.answer-box textarea {
    font-size: 1.05em !important;
    font-weight: 500 !important;
    color: #a5f3fc !important;
    background: #060f18 !important;
    border: 1px solid #0e2a40 !important;
    border-radius: 10px !important;
    line-height: 1.65 !important;
}
.conf-box textarea {
    color: #fcd34d !important;
    background: #110e00 !important;
    border: 1px solid #332200 !important;
    font-family: monospace !important;
    font-size: 0.9em !important;
}
.lat-box textarea {
    color: #6ee7b7 !important;
    background: #02100a !important;
    border: 1px solid #083d20 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82em !important;
}

textarea, input[type=text] {
    background: #0e0e1c !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
textarea:focus, input[type=text]:focus {
    border-color: var(--purple) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,.2) !important;
}

button.primary { background: linear-gradient(135deg, #7c3aed, #4f46e5) !important; border: none !important; color: #fff !important; font-weight: 600 !important; border-radius: 9px !important; }
button.primary:hover { opacity: .85 !important; }

.tab-nav button { background: transparent !important; color: var(--muted) !important; border-bottom: 2px solid transparent !important; }
.tab-nav button.selected { color: var(--cyan) !important; border-bottom: 2px solid var(--cyan) !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""

HEADER = """
<div class="header-wrap">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px">
    <span style="font-size:2.6em">🥽</span>
    <div>
      <h1 style="margin:0;font-size:1.8em;font-weight:700;
                 background:linear-gradient(90deg,#a78bfa,#38bdf8);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        QA Engine
      </h1>
      <p style="margin:3px 0 0;color:#64748b;font-size:.9em">
        Fine-tuned RoBERTa · ONNX INT8 · BM25 Retrieval
      </p>
    </div>
  </div>
  <p style="color:#94a3b8;margin:0;font-size:.92em;line-height:1.65;max-width:640px">
    Extractive question answering over a pluggable domain corpus.
    Swap the corpus and model via <code>CORPUS_PATH</code> and <code>MODEL_DIR</code> env vars.
  </p>
</div>
"""


def _stat(label: str, value: str) -> str:
    return (
        f"<span style='display:inline-block;background:rgba(124,58,237,.12);"
        f"border:1px solid rgba(124,58,237,.25);border-radius:999px;"
        f"padding:3px 12px;font-size:.8em;color:#c4b5fd;margin:2px'>"
        f"{label}: <b>{value}</b></span>"
    )


STATS = (
    _stat("passages", str(len(corpus)))
    + _stat("model", MODEL_DIR.name)
    + _stat("retrieval", "BM25 Okapi")
    + _stat("topics", str(len(TOPICS)))
)


def answer_corpus(question: str, top_k: int, show_passages: bool):
    if not question.strip():
        return "Enter a question above.", "", "", ""
    passages, scores, indices = _retrieve(question, int(top_k))
    if not passages:
        return "Corpus not loaded.", "", "", ""
    result = engine.answer(question, " ".join(passages))
    lat = result["latency"]
    lat_str = f"tokenize {lat['tokenize_ms']}ms · inference {lat['inference_ms']}ms · total {lat['total_ms']}ms"
    passages_md = ""
    if show_passages:
        parts = []
        for rank, (i, text, sc) in enumerate(zip(indices, passages, scores), 1):
            topic = corpus[i].get("topic", "").replace("_", " ").title()
            parts.append(f"**#{rank} — {topic}** &nbsp;`BM25 {sc:.3f}`\n\n> {text}")
        passages_md = "\n\n---\n\n".join(parts)
    return result["answer"], f"score {result['score']:.4f}", lat_str, passages_md


def answer_custom(question: str, context: str):
    if not question.strip():
        return "Enter a question.", "", ""
    if not context.strip():
        return "Enter a context passage.", "", ""
    result = engine.answer(question, context)
    lat = result["latency"]
    lat_str = f"tokenize {lat['tokenize_ms']}ms · inference {lat['inference_ms']}ms · total {lat['total_ms']}ms"
    return result["answer"], f"score {result['score']:.4f}", lat_str


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="QA Engine", css=CSS, theme=gr.themes.Base()) as demo:
        gr.HTML(HEADER)
        gr.HTML(f"<div style='margin:8px 0 18px'>{STATS}</div>")

        with gr.Tabs(elem_classes="tab-nav"):

            with gr.TabItem("🔍  Corpus Search"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300):
                        q1 = gr.Textbox(placeholder="Ask anything about the corpus…", lines=3, show_label=False)
                        with gr.Row():
                            top_k = gr.Slider(1, 10, value=3, step=1, label="Top-k passages")
                            show_p = gr.Checkbox(value=True, label="Show sources")
                        btn1 = gr.Button("Search", variant="primary")
                        gr.Examples(examples=EXAMPLES, inputs=[q1], label="Examples")

                    with gr.Column(scale=6, min_width=380):
                        ans1 = gr.Textbox(lines=4, show_label=False, interactive=False,
                                          elem_classes="answer-box", placeholder="Answer appears here…")
                        with gr.Row():
                            conf1 = gr.Textbox(label="Confidence", interactive=False,
                                               elem_classes="conf-box", scale=2)
                            lat1  = gr.Textbox(label="Latency", interactive=False,
                                               elem_classes="lat-box", scale=5)
                        sources = gr.Markdown()

                btn1.click(answer_corpus, [q1, top_k, show_p], [ans1, conf1, lat1, sources])
                q1.submit(answer_corpus, [q1, top_k, show_p], [ans1, conf1, lat1, sources])

            with gr.TabItem("📋  Custom Context"):
                gr.Markdown("_Paste any text as context and ask a question about it._")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4):
                        q2 = gr.Textbox(label="Question", lines=2)
                        ctx = gr.Textbox(label="Context", lines=10)
                        btn2 = gr.Button("Answer", variant="primary")
                    with gr.Column(scale=6):
                        ans2  = gr.Textbox(lines=5, show_label=False, interactive=False, elem_classes="answer-box")
                        conf2 = gr.Textbox(label="Confidence", interactive=False, elem_classes="conf-box")
                        lat2  = gr.Textbox(label="Latency", interactive=False, elem_classes="lat-box")

                btn2.click(answer_custom, [q2, ctx], [ans2, conf2, lat2])

            with gr.TabItem("ℹ️  About"):
                topic_list = "\n".join(f"- {t.replace('_', ' ').title()}" for t in TOPICS)
                gr.Markdown(f"""
### Model

| | |
|---|---|
| Base | `deepset/roberta-base-squad2` |
| Format | ONNX INT8 (dynamic quantization) |
| Size | ~120 MB (4× vs FP32) |
| Runtime | ONNX Runtime CPU / CUDA |
| Retrieval | BM25 Okapi |

### Corpus topics
{topic_list if topic_list else "_No corpus loaded_"}

### Using a different corpus

Set `CORPUS_PATH` to any JSON file with the schema:
```json
[
  {{"topic": "my_topic", "text": "passage text…"}},
  …
]
```

Set `MODEL_DIR` to point to a different fine-tuned ONNX model directory.

### Known limitations
- Extractive only — answers must appear verbatim in a passage
- BM25 misses purely semantic matches; vector retrieval is a planned upgrade
                """)

        gr.HTML(
            "<p style='text-align:center;color:#334155;font-size:.78em;margin-top:20px'>"
            "QA Engine · ONNX INT8 · BM25</p>"
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    build_ui().launch(share=args.share, server_name=args.host, server_port=args.port, show_error=True)
