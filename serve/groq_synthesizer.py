import hashlib
import os
import time
from typing import Optional

try:
    from groq import Groq
    _HAS_GROQ = True
except ImportError:
    _HAS_GROQ = False

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL_FAST   = os.getenv("GROQ_MODEL_FAST",  "llama-3.1-8b-instant")
MODEL_SMART  = os.getenv("GROQ_MODEL_SMART", "llama-3.3-70b-versatile")
MAX_TOKENS   = int(os.getenv("GROQ_MAX_TOKENS",   "400"))
TEMPERATURE  = float(os.getenv("GROQ_TEMPERATURE", "0.1"))

_MAX_PASSAGES  = 3
_MAX_PASS_CHARS = 600
_CACHE_MAX     = 500

_SYSTEM = (
    "You are a precise technical assistant. Answer the user's question using ONLY the "
    "document passages provided below. "
    "Cite the source as [Passage N] when drawing from a specific passage. "
    "If the passages do not contain sufficient information, respond with exactly: "
    "'The provided documents do not contain enough information to answer this question.' "
    "Never add information from your training data. Keep the answer concise and factual."
)

_client: Optional[object] = None
_cache:  dict[str, tuple[str, str]] = {}


def _get_client():
    global _client
    if not _HAS_GROQ or not GROQ_API_KEY:
        return None
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _cache_key(question: str, passages: list[str]) -> str:
    payload = question.lower().strip() + "||" + "||".join(
        p[:120] for p in passages[:_MAX_PASSAGES]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _is_complex(question: str, confidence: float) -> bool:
    triggers = {
        "explain", "compare", "why", "how does", "difference between",
        "elaborate", "describe", "what causes", "contrast", "trade-off",
        "pros and cons", "advantage", "disadvantage",
    }
    q = question.lower()
    return (
        len(question.split()) > 10
        or confidence < -1.5
        or any(t in q for t in triggers)
    )


def _build_user_message(question: str, passages: list[str]) -> str:
    limited = passages[:_MAX_PASSAGES]
    context = "\n\n".join(
        f"[Passage {i + 1}]\n{p[:_MAX_PASS_CHARS]}{'...' if len(p) > _MAX_PASS_CHARS else ''}"
        for i, p in enumerate(limited)
    )
    return f"Document passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def _evict_if_full() -> None:
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))


def synthesize(
    question: str,
    passages: list[str],
    confidence: float = 0.0,
) -> tuple[str | None, str]:
    client = _get_client()
    if not client or not passages:
        return None, ""

    key = _cache_key(question, passages)
    if key in _cache:
        answer, label = _cache[key]
        return answer, "cached"

    use_smart   = _is_complex(question, confidence)
    model_order = [MODEL_SMART, MODEL_FAST] if use_smart else [MODEL_FAST, MODEL_SMART]
    user_msg    = _build_user_message(question, passages)

    for model in model_order:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                text = resp.choices[0].message.content.strip()
                if not text:
                    break
                label = "groq-70b" if "70b" in model else "groq-8b"
                _evict_if_full()
                _cache[key] = (text, label)
                return text, label
            except Exception as exc:
                err = str(exc).lower()
                is_rate_limit = "rate" in err or "429" in err or "limit" in err
                if is_rate_limit and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                break

    return None, ""


def is_available() -> bool:
    return bool(_HAS_GROQ and GROQ_API_KEY)


def cache_stats() -> dict:
    return {"entries": len(_cache), "capacity": _CACHE_MAX}
