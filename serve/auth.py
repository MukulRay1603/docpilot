import os
from typing import Optional

from fastapi import Header, HTTPException

_ROLE_ORDER: dict[str, int] = {"viewer": 0, "editor": 1, "admin": 2}

REQUIRE_AUTH: bool = os.getenv("REQUIRE_AUTH", "0") == "1"

_API_KEYS: dict[str, str] = {}
_raw = os.getenv("API_KEYS", "")
if _raw:
    for pair in _raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            k, r = pair.split(":", 1)
            _API_KEYS[k.strip()] = r.strip().lower()

if not _API_KEYS:
    _API_KEYS = {
        "dp-viewer-demo": "viewer",
        "dp-editor-demo": "editor",
        "dp-admin-demo":  "admin",
    }


async def get_role(x_api_key: Optional[str] = Header(default=None)) -> str:
    if not REQUIRE_AUTH:
        return "admin"
    if not x_api_key or x_api_key not in _API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
    return _API_KEYS[x_api_key]


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
