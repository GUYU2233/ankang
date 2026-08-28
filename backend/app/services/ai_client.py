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

    def ensure_stream(self, stream_id: str, source: str, target_fps: float = 15.0) -> bool:
        """确保 AI 引擎侧存在某设备的连续流推理 worker。"""
        try:
            resp = httpx.post(
                f"{self.settings.ai_engine_url}/v1/streams/{stream_id}/start",
                json={"source": source, "target_fps": target_fps, "loop_file": True},
                timeout=8,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning(f"启动流推理失败: {exc}")
            return False

    def get_stream_latest(self, stream_id: str) -> AIInferResponse | None:
        try:
            resp = httpx.get(
                f"{self.settings.ai_engine_url}/v1/streams/{stream_id}/latest",
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            result = data.get("result")
            return AIInferResponse(**result) if result else None
        except Exception as exc:
            logger.warning(f"读取流结果失败: {exc}")
            return None

    def get_stream_frame(self, stream_id: str) -> bytes | None:
        """获取 AI 引擎侧最新一帧的带骨架标注 JPEG。"""
        try:
            resp = httpx.get(
                f"{self.settings.ai_engine_url}/v1/streams/{stream_id}/frame.jpg",
                timeout=8,
            )
            if resp.status_code == 200 and resp.content:
                return resp.content
        except Exception as exc:
            logger.warning(f"读取流画面失败: {exc}")
        return None

    def stop_stream(self, stream_id: str) -> None:
        try:
            httpx.post(f"{self.settings.ai_engine_url}/v1/streams/{stream_id}/stop", timeout=8)
        except Exception:
            pass

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