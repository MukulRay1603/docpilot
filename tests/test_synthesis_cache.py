"""Fix 8: the Groq answer cache is LRU, and the README no longer claims a
confidence gate the code doesn't have."""
from pathlib import Path

from serve import groq_synthesizer as g

ROOT = Path(__file__).resolve().parent.parent


def test_cache_is_lru(monkeypatch):
    monkeypatch.setattr(g, "_CACHE_MAX", 2)
    g._cache.clear()
    g._cache_put("a", ("A", "groq-8b"))
    g._cache_put("b", ("B", "groq-8b"))
    assert g._cache_get("a") == ("A", "groq-8b")  # hit refreshes "a"
    g._cache_put("c", ("C", "groq-8b"))            # evicts least-recently-used "b"
    assert g._cache_get("b") is None
    assert g._cache_get("a") is not None and g._cache_get("c") is not None
    assert len(g._cache) == 2
    g._cache.clear()


def test_cache_key_sees_full_prompt_passage_text():
    base = "x" * 120
    assert g._cache_key("q", [base + " alpha"]) != g._cache_key("q", [base + " beta"])


def test_synthesize_returns_cached_on_hit(monkeypatch):
    monkeypatch.setattr(g, "_get_client", lambda: object())
    g._cache.clear()
    key = g._cache_key("q", ["p"])
    g._cache_put(key, ("answer", "groq-8b"))
    assert g.synthesize("q", ["p"]) == ("answer", "cached")
    g._cache.clear()


def test_readme_matches_code():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "confidence-gated" not in readme.lower()
    assert "High confidence → return extractive span" not in readme
