"""
API key authentication and role-based access control.

Three roles:
    viewer  — query endpoints only
    editor  — viewer + corpus ingest
    admin   — editor + audit logs + security events

Set REQUIRE_AUTH=1 to enforce key checks (off by default for local dev).
Provide keys via: API_KEYS=key1:viewer,key2:editor,key3:admin

Demo keys active when API_KEYS is not set (safe for local use only).
"""

import os
from typing import Optional

from fastapi import Header, HTTPException

# Role ordering: higher index = more privileged
_ROLE_ORDER: dict[str, int] = {"viewer": 0, "editor": 1, "admin": 2}

REQUIRE_AUTH: bool = os.getenv("REQUIRE_AUTH", "0") == "1"

# Parse API_KEYS env var: "keyA:viewer,keyB:admin"
_API_KEYS: dict[str, str] = {}

_raw = os.getenv("API_KEYS", "")
if _raw:
    for pair in _raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            k, r = pair.split(":", 1)
            _API_KEYS[k.strip()] = r.strip().lower()

# Fall back to demo keys (only used when REQUIRE_AUTH=0 anyway)
if not _API_KEYS:
    _API_KEYS = {
        "dp-viewer-demo": "viewer",
        "dp-editor-demo": "editor",
        "dp-admin-demo":  "admin",
    }


async def get_role(x_api_key: Optional[str] = Header(default=None)) -> str:
    """FastAPI dependency — returns the caller's role string."""
    if not REQUIRE_AUTH:
        return "admin"  # open access for local dev / demo
    if not x_api_key or x_api_key not in _API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
    return _API_KEYS[x_api_key]


def require_role(minimum: str):
    """
    Dependency factory.  Usage:

        @app.get("/audit/logs")
        def logs(role: str = Depends(require_role("admin"))):
            ...
    """
    async def _dep(x_api_key: Optional[str] = Header(default=None)) -> str:
        role = await get_role(x_api_key)
        if _ROLE_ORDER.get(role, -1) < _ROLE_ORDER.get(minimum, 0):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' is not permitted here (need '{minimum}')",
            )
        return role
    return _dep


def list_demo_keys() -> dict[str, str]:
    """Return demo key → role mapping (for /health display)."""
    return {k: v for k, v in _API_KEYS.items() if k.startswith("dp-") and k.endswith("-demo")}
