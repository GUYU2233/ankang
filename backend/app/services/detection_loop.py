from __future__ import annotations

import asyncio
import time

from loguru import logger
from sqlalchemy import select

from app.config import get_settings
from app.core.alert_engine import AlertEngine
from app.core.notify import manager
from app.core.stream_service import stream_service
from app.db import SessionLocal
from app.models.entities import Device
from app.services.ai_client import ai_client


class DetectionLoop:
    """端到端巡检循环：取流 -> AI 推理 -> 风险评分 -> 分级预警 -> 推送。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.alert_engine = AlertEngine(confirm_frames=self.settings.alert_confirm_frames)
        self._stop = False

    async def run_forever(self) -> None:
        logger.info("检测巡检循环已启动")
        while not self._stop:
            started = time.time()
            try:
                await self._tick()
            except Exception as exc:
                logger.exception(f"巡检循环异常: {exc}")
            await asyncio.sleep(max(0.5, self.settings.detect_interval_seconds - (time.time() - started)))

    def stop(self) -> None:
        self._stop = True

    async def _tick(self) -> None:
        with SessionLocal() as db:
            devices = db.scalars(select(Device).where(Device.enabled == True)).all()
        for device in devices:
            try:
                broadcast_msg = await asyncio.to_thread(self._process_device, device.id)
            except Exception as exc:
                logger.warning(f"设备 {device.id} 处理失败: {exc}")
                continue
            if broadcast_msg:
                await manager.broadcast({"type": "alert", "data": broadcast_msg})

    def _is_stream_source(self, device: Device) -> bool:
        url = (device.access_url or "").strip()
        if not url:
            return False
        return url.lower().startswith("rtsp://") or url.lower().endswith((".mp4", ".avi", ".mkv"))

    def _process_device(self, device_id: int):
        with SessionLocal() as db:
            device = db.get(Device, device_id)
            if device is None:
                return None
            stream_id = f"device-{device.id}"
            if self._is_stream_source(device):
                url = (device.access_url or "").strip()
                if url.lower().endswith((".mp4", ".avi", ".mkv")):
                    url = stream_service.resolve_video_path(url)
                ai_client.ensure_stream(stream_id, url, target_fps=15.0)
                infer = ai_client.get_stream_latest(stream_id)
                if infer is None:
                    packet = stream_service.get_frame(device)
                    infer = ai_client.infer_frame(packet.frame, stream_id=stream_id)
            else:
                packet = stream_service.get_frame(device)
                infer = ai_client.infer_frame(packet.frame, stream_id=stream_id)
            if infer is None:
                packet = stream_service.get_frame(device)
                infer = stream_service.build_demo_result(device, packet.meta)
            stream_service.set_latest_result(device.id, infer)
            return self.alert_engine.process(db, device, infer, infer.risk_score, time.time())