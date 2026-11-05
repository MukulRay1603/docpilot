"""Fix 7: auth on /security/*, no implicit demo keys, constant-time compare."""
import pytest
from fastapi.testclient import TestClient

from serve import auth


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIT_DB", str(tmp_path / "audit.db"))
    monkeypatch.setattr(auth, "REQUIRE_AUTH", True)
    monkeypatch.setattr(auth, "_KEY_DIGESTS", auth.parse_keys("s3cret-admin:admin,s3cret-view:viewer"))
    import serve.app as app_mod
    return TestClient(app_mod.app)


@pytest.mark.parametrize("path,body", [
    ("/security/check", {"text": "hello"}),
    ("/security/check-document", {"text": "hello"}),
    ("/security/redact", {"text": "a@b.com"}),
])
def test_security_endpoints_require_key(client, path, body):
    assert client.post(path, json=body).status_code == 401
    assert client.post(path, json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post(path, json=body, headers={"X-API-Key": "s3cret-view"}).status_code == 200


def test_demo_keys_off_by_default(client):
    for k in ("dp-viewer-demo", "dp-editor-demo", "dp-admin-demo"):
        assert client.get("/corpus/stats", headers={"X-API-Key": k}).status_code == 401
    assert auth.parse_keys("") == []


def test_demo_keys_only_with_explicit_demo_mode():
    digests = auth.parse_keys("", demo_mode=True)
    assert auth.lookup_role("dp-admin-demo", digests) == "admin"


def test_roles_still_enforced(client):
    assert client.get("/audit/stats", headers={"X-API-Key": "s3cret-view"}).status_code == 403
    assert client.get("/audit/stats", headers={"X-API-Key": "s3cret-admin"}).status_code == 200


def test_constant_time_lookup_and_no_plaintext(monkeypatch):
    calls = []
    real = auth.hmac.compare_digest
    monkeypatch.setattr(auth.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    digests = auth.parse_keys("k1:viewer,k2:editor,k3:admin")
    assert auth.lookup_role("k1", digests) == "viewer"
    assert len(calls) == 3  # no early exit on match
    assert all(isinstance(d, bytes) and len(d) == 32 for d, _ in digests)
    assert auth.lookup_role("nope", digests) is None
    assert auth.parse_keys("k:superuser") == []  # unknown roles rejected
