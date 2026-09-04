"""多模态视觉巡检运行配置（单例，DB 持久化 + 环境变量兜底）。"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.secrets import decrypt_secret, encrypt_secret
from app.models.entities import MultimodalConfigRecord


@dataclass
class RuntimeConfig:
    provider: str = "qwen"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    enabled: bool = False
    interval_seconds: int = 60
    temperature: float = 0.2
    max_tokens: int = 800
    timeout_seconds: float = 30.0
    prompt_override: str | None = None

    def mask_key(self) -> str:
        k = self.api_key or ""
        if not k:
            return ""
        if len(k) <= 8:
            return "*" * len(k)
        return k[:4] + "*" * (len(k) - 8) + k[-4:]

    def is_ready(self) -> bool:
        """是否具备可调用条件（启用且有模型/密钥或环境变量兜底）。"""
        if not self.enabled:
            return False
        if not self.model and not self.base_url:
            # 未显式配置时，靠提供商预设默认模型与 API Key 环境变量即可
            return self.provider != "custom"
        return True


class MultimodalConfigService:
    def __init__(self) -> None:
        s = get_settings()
        self._runtime = RuntimeConfig(
            provider=s.multimodal_provider,
            model=s.multimodal_model,
            base_url=s.multimodal_base_url,
            api_key=s.multimodal_api_key,
            enabled=s.multimodal_enabled,
            interval_seconds=s.multimodal_interval_seconds,
            temperature=s.multimodal_temperature,
            max_tokens=s.multimodal_max_tokens,
            timeout_seconds=s.multimodal_timeout_seconds,
        )

    def current(self) -> RuntimeConfig:
        return self._runtime

    def _from_record(self, rec: MultimodalConfigRecord) -> RuntimeConfig:
        return RuntimeConfig(
            provider=rec.provider,
            model=rec.model,
            base_url=rec.base_url,
            api_key=decrypt_secret(rec.api_key, "multimodal.api_key"),
            enabled=rec.enabled,
            interval_seconds=rec.interval_seconds,
            temperature=rec.temperature,
            max_tokens=rec.max_tokens,
            timeout_seconds=self._runtime.timeout_seconds,
            prompt_override=rec.prompt_override,
        )

    def reload(self, db: Session) -> RuntimeConfig:
        """从 DB 读取配置；不存在则用环境变量默认值新建。"""
        rec = db.get(MultimodalConfigRecord, 1)
        if rec is None:
            seeds = self._runtime
            rec = MultimodalConfigRecord(
                id=1,
                provider=seeds.provider,
                model=seeds.model,
                base_url=seeds.base_url,
                api_key=seeds.api_key,
                enabled=seeds.enabled,
                interval_seconds=seeds.interval_seconds,
                temperature=seeds.temperature,
                max_tokens=seeds.max_tokens,
            )
            db.add(rec)
            db.commit()
            db.refresh(rec)
        self._runtime = self._from_record(rec)
        return self._runtime

    def update(self, db: Session, **fields) -> RuntimeConfig:
        """按字段更新单行配置（id=1）。"""
        rec = db.get(MultimodalConfigRecord, 1)
        if rec is None:
            rec = MultimodalConfigRecord(id=1)
            db.add(rec)
        for k, v in fields.items():
            if hasattr(rec, k):
                if k == "api_key" and v:
                    v = encrypt_secret(v, "multimodal.api_key")
                setattr(rec, k, v)
        db.commit()
        db.refresh(rec)
        self._runtime = self._from_record(rec)
        return self._runtime


_config_service: MultimodalConfigService | None = None


def get_multimodal_config_service() -> MultimodalConfigService:
    global _config_service
    if _config_service is None:
        _config_service = MultimodalConfigService()
    return _config_service
