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

    def _process_device(self, device_id: int):
        with SessionLocal() as db:
            device = db.get(Device, device_id)
            if device is None:
                return None
            packet = stream_service.get_frame(device)
            infer = ai_client.infer_frame(packet.frame)
            if infer is None:
                infer = stream_service.build_demo_result(device, packet.meta)
            return self.alert_engine.process(db, device, infer, infer.risk_score, time.time())