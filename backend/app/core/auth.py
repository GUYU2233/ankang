"""轻量 API Key 鉴权：生产强制，开发默认关闭以保持本地兼容。"""
from __future__ import annotations

import hashlib
import hmac

from fastapi import Header, HTTPException

from app.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if settings.auth_mode == "disabled":
        return
    expected = settings.api_key
    if not expected:
        raise HTTPException(status_code=503, detail="服务端未配置 API_KEY")
    supplied = x_api_key or ""
    if not hmac.compare_digest(hashlib.sha256(supplied.encode()).digest(), hashlib.sha256(expected.encode()).digest()):
        raise HTTPException(status_code=401, detail="无效的 API Key")
