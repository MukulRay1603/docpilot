"""
Local LLM synthesis layer via Ollama.

Why Ollama instead of a cloud API:
- The whole point of DocPilot is to work on internal company document filesystems.
  Nothing leaves the machine. No API costs, no data privacy risk.
- Ollama runs open-weight models locally (Llama 3.2, Mistral, Qwen, etc.)
- Falls back silently to the extractive answer if Ollama isn't running.

Setup (one time):
    # Windows: download from https://ollama.ai or: winget install Ollama.Ollama
    # Pull a model (3B is fast on CPU/GPU, 7B is better quality):
    ollama pull llama3.2        # ~2GB, fits in 6GB VRAM easily
    # or: ollama pull mistral   # ~4GB, better reasoning

Usage:
    Set OLLAMA_MODEL env var to switch models. Default is llama3.2.
    Set SYNTHESIS_TIMEOUT env var to adjust timeout (default 30s).
"""

import os
import httpx

OLLAMA_URL     = os.getenv("OLLAMA_URL",      "http://localhost:11434/api/generate")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",    "llama3.2")
SYNTHESIS_TIMEOUT = float(os.getenv("SYNTHESIS_TIMEOUT", "30"))

# System prompt: forces the model to stay grounded in the retrieved passages.
# "Do not use outside knowledge" is important for a company doc system where
# hallucinated answers are worse than "I don't know."
_SYSTEM = (
    "You are a technical support assistant. Answer the user's question using "
    "ONLY the document passages provided below. "
    "If the passages do not contain enough information to answer, respond with exactly: "
    "'The provided documents do not contain enough information to answer this question.' "
    "Do not add information from your training data. Be concise."
)


def synthesize(question: str, passages: list[str]) -> str | None:
    """
    Returns a synthesised answer grounded in `passages`, or None if Ollama
    is not reachable. Callers should fall back to the extractive answer on None.
    """
    if not passages:
        return None

    context = "\n\n".join(
        f"[Passage {i + 1}]\n{p.strip()}" for i, p in enumerate(passages)
    )
    prompt = f"{_SYSTEM}\n\nDocument passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300},
            },
            timeout=SYNTHESIS_TIMEOUT,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        return answer if answer else None
    except httpx.ConnectError:
        # Ollama not running -- silently degrade to extractive
        return None
    except Exception:
        return None


def is_available() -> bool:
    """Quick check whether Ollama is up."""
    try:
        httpx.get("http://localhost:11434", timeout=2.0)
        return True
    except Exception:
        return False
