"""Optional production dependencies (API key gate)."""

from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key_if_configured(x_api_key: str | None = Security(_api_key_header)) -> None:
    raw = (settings.api_keys or "").strip()
    if not raw:
        return
    allowed = {k.strip() for k in raw.split(",") if k.strip()}
    if not x_api_key or x_api_key not in allowed:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
