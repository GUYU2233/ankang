from __future__ import annotations

import cv2
import httpx
from loguru import logger

from app.config import get_settings
from app.schemas.schemas import AIInferResponse


class AIEngineClient:
    """调用 AI 推理引擎（ai-engine 服务）。"""

    def __init__(self) -> None:
        self.settings = get_settings()

    def infer_frame(self, frame, stream_id: str = "default") -> AIInferResponse | None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        try:
            resp = httpx.post(
                f"{self.settings.ai_engine_url}/v1/infer",
                files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                params={"stream_id": stream_id},
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            return AIInferResponse(**data)
        except Exception as exc:
            logger.warning(f"AI 引擎调用失败，使用本地模拟兜底: {exc}")
            return None


ai_client = AIEngineClient()