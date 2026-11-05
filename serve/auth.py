import hashlib
import hmac
import os
import sys
from typing import Optional

from fastapi import Header, HTTPException

_ROLE_ORDER: dict[str, int] = {"viewer": 0, "editor": 1, "admin": 2}

REQUIRE_AUTH: bool = os.getenv("REQUIRE_AUTH", "0") == "1"
# Demo keys are public (they are in this file). They are only accepted when
# DEMO_MODE=1 is set explicitly, never as a silent fallback.
DEMO_MODE: bool = os.getenv("DEMO_MODE", "0") == "1"
_DEMO_KEYS = {
    "dp-viewer-demo": "viewer",
    "dp-editor-demo": "editor",
    "dp-admin-demo":  "admin",
}


def _digest(key: str) -> bytes:
    return hashlib.sha256(key.encode("utf-8")).digest()


def parse_keys(raw: str, demo_mode: bool = False) -> list[tuple[bytes, str]]:
    """Parse API_KEYS="key:role,key2:role2" into (sha256(key), role) pairs.
    Only digests are kept in memory."""
    keys: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            k, r = pair.split(":", 1)
            r = r.strip().lower()
            if k.strip() and r in _ROLE_ORDER:
                keys[k.strip()] = r
    if demo_mode:
        keys = {**_DEMO_KEYS, **keys}
    return [(_digest(k), r) for k, r in keys.items()]


_KEY_DIGESTS: list[tuple[bytes, str]] = parse_keys(os.getenv("API_KEYS", ""), DEMO_MODE)

if REQUIRE_AUTH and not _KEY_DIGESTS:
    print("[auth] REQUIRE_AUTH=1 but no API_KEYS set (and DEMO_MODE is off): "
          "every protected request will get 401", file=sys.stderr)


def lookup_role(api_key: str, key_digests: list[tuple[bytes, str]]) -> Optional[str]:
    """Constant-time lookup: compares the candidate's digest against every
    stored digest with hmac.compare_digest, without early exit."""
    candidate = _digest(api_key)
    found: Optional[str] = None
    for digest, role in key_digests:
        if hmac.compare_digest(candidate, digest):
            found = role
    return found


async def get_role(x_api_key: Optional[str] = Header(default=None)) -> str:
    if not REQUIRE_AUTH:
        return "admin"
    role = lookup_role(x_api_key, _KEY_DIGESTS) if x_api_key else None
    if role is None:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
    return role


def require_role(minimum: str):
    async def _dep(x_api_key: Optional[str] = Header(default=None)) -> str:
        role = await get_role(x_api_key)
        if _ROLE_ORDER.get(role, -1) < _ROLE_ORDER.get(minimum, 0):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' is not permitted here (need '{minimum}')",
            )
        return role
    return _dep
