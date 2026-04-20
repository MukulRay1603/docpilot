import json
from pathlib import Path

import gradio as gr

from config import MODEL_DIR, CORPUS_PATH, SCORE_THRESHOLD
from serve.inference import QAEngine
from serve.retrieval import Retriever
from serve.synthesizer import synthesize, is_available as ollama_up
from ingest import ingest_file

engine = QAEngine(MODEL_DIR)

corpus: list[dict] = []
retriever: Retriever = Retriever([])

if CORPUS_PATH.exists():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    retriever = Retriever(corpus)
    print(f"Loaded {len(corpus)} passages from {CORPUS_PATH.name}")

EXAMPLES = [
    ["How does inside-out tracking work?"],
    ["What is the vergence-accommodation conflict?"],
    ["How does INT8 quantization reduce latency?"],
    ["What causes simulator sickness in VR?"],
    ["How does WebRTC handle packet loss in XR?"],
    ["What is asynchronous timewarp?"],
    ["How are annotations anchored in remote assist?"],
]

CSS = """
:root {
    --bg: #0d0d1a; --card: #12121f; --border: #1e1e35;
    --purple: #7c3aed; --indigo: #4f46e5; --cyan: #06b6d4;
    --text: #e2e8f0; --muted: #64748b;
}
body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; }
.gr-button-primary { background: var(--purple) !important; border: none !important; }
.gr-box, .gr-panel { background: var(--card) !important; border-color: var(--border) !important; }
"""


# ---------------------------------------------------------------------------
# Tab 1 -- Corpus search
# ---------------------------------------------------------------------------

def search_and_answer(question: str, top_k: int) -> tuple[str, str, str]:
    if not question.strip():
        return "", "", ""
    if not corpus:
        return "No corpus loaded. Use the 'Your Corpus' tab to add documents.", "", ""

    passages = retriever.retrieve(question, top_k=int(top_k))
    if not passages:
        return "No passages retrieved.", "", ""

    context = " ".join(p["text"] for p in passages)
    result = engine.answer(question, context)

    confident = result["score"] >= SCORE_THRESHOLD
    answer = result["answer"] if (confident and result["answer"]) else "No confident answer found in the corpus."

    sources = list({p.get("source") or p.get("topic") or "unknown" for p in passages})
    lat = result["latency"]
    meta = f"Sources: {', '.join(sources)}\nLatency: {lat['total_ms']} ms  (tok {lat['tokenize_ms']} + inf {lat['inference_ms']})"

    context_preview = "\n\n---\n\n".join(
        p["text"][:400] + "..." if len(p["text"]) > 400 else p["text"]
        for p in passages
    )
    return answer, meta, context_preview


# ---------------------------------------------------------------------------
# Tab 2 -- Custom context
# ---------------------------------------------------------------------------

def answer_custom(question: str, context: str) -> tuple[str, str]:
    if not question.strip() or not context.strip():
        return "", ""
    result = engine.answer(question, context)
    confident = result["score"] >= SCORE_THRESHOLD
    answer = result["answer"] if (confident and result["answer"]) else "Not enough information in the provided context."
    return answer, f"Latency: {result['latency']['total_ms']} ms"


# ---------------------------------------------------------------------------
# Tab 3 -- Your Corpus
# ---------------------------------------------------------------------------

def get_corpus_stats() -> str:
    if not corpus:
        return "Corpus is empty."
    topics: dict[str, int] = {}
    for p in corpus:
        key = p.get("topic") or p.get("source") or "unknown"
        topics[key] = topics.get(key, 0) + 1
    lines = [f"Total passages: {len(corpus)}", ""]
    for t, n in sorted(topics.items()):
        lines.append(f"  {t}: {n}")
    return "\n".join(lines)


def add_text_to_corpus(text: str, source_label: str) -> tuple[str, str]:
    global corpus, retriever
    if not text.strip():
        return "Nothing to add.", get_corpus_stats()
    label = source_label.strip() or "upload"
    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 80]
    if not chunks:
        return "No usable chunks. Separate sections with blank lines (each >80 chars).", get_corpus_stats()
    start_id = max((p["id"] for p in corpus), default=-1) + 1
    new_passages = [
        {"id": start_id + i, "source": label, "topic": label, "text": chunk}
        for i, chunk in enumerate(chunks)
    ]
    corpus.extend(new_passages)
    retriever.rebuild(corpus)
    return f"Added {len(new_passages)} passages from '{label}' (total: {len(corpus)}).", get_corpus_stats()


def add_file_to_corpus(file, source_label: str) -> tuple[str, str]:
    global corpus, retriever
    if file is None:
        return "No file selected.", get_corpus_stats()
    path = Path(file.name)
    label = source_label.strip() or path.stem
    try:
        new_passages = ingest_file(path)
    except ImportError as e:
        return f"Missing dependency: {e}", get_corpus_stats()
    except Exception as e:
        return f"Error reading file: {e}", get_corpus_stats()
    if not new_passages:
        return "No usable content found in the file.", get_corpus_stats()
    start_id = max((p["id"] for p in corpus), default=-1) + 1
    for i, p in enumerate(new_passages):
        p["id"] = start_id + i
        p["source"] = label
        p["topic"] = label
    corpus.extend(new_passages)
    retriever.rebuild(corpus)
    return f"Added {len(new_passages)} passages from '{path.name}' (total: {len(corpus)}).", get_corpus_stats()


def reset_to_demo() -> tuple[str, str]:
    global corpus, retriever
    if CORPUS_PATH.exists():
        corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        retriever = Retriever(corpus)
        return f"Reset to demo corpus ({len(corpus)} passages).", get_corpus_stats()
    return "Demo corpus not found. Run: python data/build_demo_corpus.py", get_corpus_stats()


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

with gr.Blocks(css=CSS, title="DocPilot") as demo:
    gr.Markdown(
        "# DocPilot\n"
        "Document QA over your own corpus. "
        "Fine-tuned RoBERTa + BM25 retrieval + ONNX INT8 inference."
    )

    with gr.Tab("Search Corpus"):
        with gr.Row():
            with gr.Column(scale=2):
                q1 = gr.Textbox(label="Question", placeholder="What does INT8 quantization do to latency?", lines=2)
                top_k = gr.Slider(1, 8, value=3, step=1, label="Passages to retrieve")
                btn1 = gr.Button("Ask", variant="primary")
            with gr.Column(scale=3):
                ans1 = gr.Textbox(label="Answer", lines=3, interactive=False)
                meta1 = gr.Textbox(label="Source / Latency", lines=2, interactive=False)
        ctx_preview = gr.Textbox(label="Retrieved passages", lines=8, interactive=False)
        gr.Examples(examples=EXAMPLES, inputs=[q1])
        btn1.click(search_and_answer, inputs=[q1, top_k], outputs=[ans1, meta1, ctx_preview])

    with gr.Tab("Custom Context"):
        gr.Markdown("Paste any text and ask a question against it directly, without corpus retrieval.")
        with gr.Row():
            ctx2 = gr.Textbox(label="Context", lines=10, placeholder="Paste your document text here...")
            with gr.Column():
                q2 = gr.Textbox(label="Question", lines=2)
                btn2 = gr.Button("Ask", variant="primary")
                ans2 = gr.Textbox(label="Answer", lines=3, interactive=False)
                meta2 = gr.Textbox(label="Latency", lines=1, interactive=False)
        btn2.click(answer_custom, inputs=[q2, ctx2], outputs=[ans2, meta2])

    with gr.Tab("Your Corpus"):
        gr.Markdown(
            "Add your own documents. The retrieval index rebuilds in memory after each addition.\n\n"
            "For persistent changes run `python ingest.py --source your_docs/ --output data/corpus.json`."
        )
        stats_box = gr.Textbox(label="Current corpus", value=get_corpus_stats, lines=8, interactive=False)

        with gr.Accordion("Add text directly", open=True):
            source_label_text = gr.Textbox(label="Source label", value="my_docs")
            text_input = gr.Textbox(
                label="Paste text (blank line between sections)",
                lines=10,
                placeholder="Each paragraph separated by a blank line becomes one searchable passage."
            )
            add_text_btn = gr.Button("Add to corpus")
            text_status = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Accordion("Upload a file (PDF, TXT, MD)", open=False):
            source_label_file = gr.Textbox(label="Source label", placeholder="defaults to filename")
            file_upload = gr.File(label="Upload", file_types=[".pdf", ".txt", ".md", ".rst"])
            add_file_btn = gr.Button("Add file to corpus")
            file_status = gr.Textbox(label="Status", interactive=False, lines=1)

        reset_btn = gr.Button("Reset to demo corpus", variant="secondary")
        reset_status = gr.Textbox(label="Status", interactive=False, lines=1)

        add_text_btn.click(add_text_to_corpus, inputs=[text_input, source_label_text], outputs=[text_status, stats_box])
        add_file_btn.click(add_file_to_corpus, inputs=[file_upload, source_label_file], outputs=[file_status, stats_box])
        reset_btn.click(reset_to_demo, outputs=[reset_status, stats_box])

    with gr.Tab("About"):
        gr.Markdown("""
## DocPilot

Built as a remote support tool for an XR company where field technicians needed to query device
manuals and maintenance procedures hands-free while working. The corpus was internal technical
documentation; this repo ships with a synthetic demo corpus covering the same domain.

The inference stack targets low-latency deployment on GPU-constrained hardware (XR headsets),
which is why the model path goes through ONNX export and INT8 quantization rather than serving
PyTorch directly. On CPU, INT8 is roughly 3x faster than FP32 at the same quality.

**Model**: `deepset/roberta-base-squad2` fine-tuned on a domain-specific corpus.
Extractive QA -- answers are spans extracted directly from retrieved passages, not generated text.

**Retrieval**: BM25 by default. Set `USE_SEMANTIC=1` and `pip install sentence-transformers`
for hybrid BM25 + dense retrieval (better for paraphrase queries, ~200ms extra startup).

**Bring your own corpus**: Use the 'Your Corpus' tab to add documents in the browser,
or run `python ingest.py --source your_docs/` to build corpus.json from a folder of PDFs or text files.

**Known limitations**:
- Extractive only: cannot synthesise answers across multiple passages
- BM25 alone misses synonyms; enable semantic retrieval for better recall
- Demo corpus has 43 passages covering XR topics; real deployments need their own documents
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
