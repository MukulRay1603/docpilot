"""UI page: self-contained, honest copy, and wired to the real endpoints."""
import re
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")


def test_no_external_resources():
    # Must work offline: no CDN scripts, stylesheets or fonts.
    assert not re.search(r'<(script|link)[^>]+(src|href)="https?://', HTML)
    assert "cdn." not in HTML and "fonts.googleapis" not in HTML


def test_copy_rules():
    assert "—" not in HTML  # no em dashes
    low = HTML.lower()
    for claim in ("production-ready", "enterprise", "10+", "71.1", "67.0", "120 mb", "confidence-gated"):
        assert claim not in low, claim
    assert "pattern-based" in low


def test_uses_existing_endpoints_and_modes():
    for ep in ("/answer/corpus", "/health", "/security/check", "/security/check-document",
               "/security/redact", "/audit/stats", "/audit/logs", "/audit/security"):
        assert f'"{ep}' in HTML, ep
    for mode in ("bm25", "hybrid", "hybrid+rerank"):
        assert f'"{mode}"' in HTML
    assert "X-API-Key" in HTML
    assert 'role="tab"' in HTML.replace("role=tab", 'role="tab"')


def test_health_reports_engine_label(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIT_DB", str(tmp_path / "a.db"))
    import serve.app as app_mod

    class StandIn:
        label = "STAND-IN"

    monkeypatch.setattr(app_mod, "engine", StandIn())
    c = TestClient(app_mod.app)
    assert c.get("/health").json()["qa_engine_label"] == "STAND-IN"
    monkeypatch.setattr(app_mod, "engine", None)
    h = c.get("/health").json()
    assert h["qa_engine_label"] is None and h["model_loaded"] is False
    assert c.get("/").status_code == 200
