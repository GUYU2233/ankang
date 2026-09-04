"""敏感配置的版本化静态加密。根密钥来自 SECRET_ENCRYPTION_KEY。"""
from __future__ import annotations

import base64
import hashlib
import os
from functools import lru_cache


def _raw_key() -> bytes:
    value = os.getenv("SECRET_ENCRYPTION_KEY", "").strip()
    if value:
        try:
            key = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        except Exception:
            key = bytes.fromhex(value)
        if len(key) == 32:
            return key
        raise RuntimeError("SECRET_ENCRYPTION_KEY 必须解码为 32 字节")
    # 兼容本地开发；生产环境必须显式注入根密钥。
    seed = os.getenv("AUTH_SECRET", "zhihu-ankang-local-dev-only")
    return hashlib.sha256(seed.encode()).digest()


def encrypt_secret(value: str | None, aad: str = "config") -> str:
    if not value or str(value).startswith("v1."):
        return value or ""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        # 缺少 cryptography 时不允许伪加密写入。
        raise RuntimeError("请安装 cryptography 以启用敏感配置加密")
    nonce = os.urandom(12)
    data = AESGCM(_raw_key()).encrypt(nonce, str(value).encode(), aad.encode())
    return "v1." + base64.urlsafe_b64encode(nonce + data).decode()


def decrypt_secret(value: str | None, aad: str = "config") -> str:
    if not value:
        return ""
    if not str(value).startswith("v1."):
        return str(value)  # 兼容旧库明文，下一次保存会迁移
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    raw = base64.urlsafe_b64decode(str(value)[3:])
    return AESGCM(_raw_key()).decrypt(raw[:12], raw[12:], aad.encode()).decode()
