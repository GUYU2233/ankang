"""多模态视觉巡检循环：定时截图 -> 大模型识别 -> 存证 -> 分级告警。

与 DetectionLoop 并行运行，周期可配置；也支持通过 REST 手动触发单设备识别。
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
from loguru import logger
from sqlalchemy import select

from app.config import get_settings
from app.core.notify import manager
from app.core.stream_service import stream_service
from app.db import SessionLocal
from app.models.entities import (
    AlertEvent,
    AlertLevel,
    Device,
    FallEvent,
    MonitoringSnapshot,
    NotificationLog,
    Resident,
)
from app.services.multimodal_client import MultimodalError, analyze_image
from app.services.multimodal_config import RuntimeConfig, get_multimodal_config_service

SNAPSHOT_DIR = Path("runtime/multimodal")

_TITLE_BY_EVENT = {
    "fall": "{scene} 多模态识别疑似跌倒",
    "posture_abnormal": "{scene} 老人姿态/行为异常",
    "floor_clutter": "{scene} 地面杂物/跌倒隐患",
    "other_risk": "{scene} 发现其他安全风险",
}
_EVENT_TYPE_BY = {
    "fall": "fall_event",
    "posture_abnormal": "behavior_risk",
    "floor_clutter": "behavior_risk",
    "other_risk": "behavior_risk",
}


def _save_snapshot(frame, device_id: int) -> str:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"dev{device_id}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def _emit_alert(db, device: Device, result: dict, snapshot_id: int, latency_ms: int, snapshot_path: str) -> dict:
    """把多模态识别结果落库为告警，返回用于 WebSocket 广播的字典。"""
    now = datetime.now()
    event_type = _EVENT_TYPE_BY.get(result["event_type"], "behavior_risk")
    scene = device.scene or "监控区域"
    title = _TITLE_BY_EVENT.get(result["event_type"], "{scene} 多模态风险提醒").format(scene=scene)
    level = AlertLevel(result["level"])
    resident_id = device.resident_id

    alert = AlertEvent(
        alert_no=uuid.uuid4().hex[:16],
        device_id=device.id,
        resident_id=resident_id,
        level=level,
        event_type=event_type,
        title=title,
        detail_json=json.dumps(
            {
                "source": "multimodal",
                "event_type": result["event_type"],
                "severity": result["severity"],
                "confidence": result["confidence"],
                "summary": result["summary"],
                "details": result["details"],
                "location_hint": result["location_hint"],
                "suggestion": result["suggestion"],
                "person_count": result["person_count"],
                "latency_ms": latency_ms,
                "snapshot_id": snapshot_id,
                "snapshot_path": snapshot_path,
            },
            ensure_ascii=False,
        ),
        confirmed=False,
        handled=False,
        created_at=now,
    )
    db.add(alert)
    db.flush()

    db.add(NotificationLog(alert_id=alert.id, channel="websocket", target="家属端", content=title, status="sent"))

    if result["event_type"] == "fall":
        db.add(
            FallEvent(
                device_id=device.id,
                resident_id=resident_id,
                alert_id=alert.id,
                start_at=now,
                fall_prob=result["confidence"],
                screenshot_path=snapshot_path,
                note=result["summary"],
            )
        )

    resident = db.get(Resident, resident_id) if resident_id else None
    is_fall = result["event_type"] == "fall"
    return {
        "id": alert.id,
        "alert_no": alert.alert_no,
        "level": level.value,
        "event_type": event_type,
        "title": title,
        "device_id": device.id,
        "device_name": device.device_name,
        "scene": scene,
        "resident_id": resident_id,
        "resident_name": resident.name if resident else "",
        "score": round(result["confidence"], 2),
        "fall_prob": round(result["confidence"], 2) if is_fall else 0.0,
        "created_at": now.isoformat(),
        "source": "multimodal",
        "snapshot_id": snapshot_id,
    }


def _snapshot_dict(snp: MonitoringSnapshot) -> dict:
    return {
        "id": snp.id,
        "device_id": snp.device_id,
        "resident_id": snp.resident_id,
        "provider": snp.provider,
        "model": snp.model,
        "event_type": snp.event_type,
        "severity": snp.severity,
        "confidence": snp.confidence,
        "has_issue": snp.has_issue,
        "level": snp.level.value if isinstance(snp.level, AlertLevel) else str(snp.level),
        "summary": snp.summary,
        "detail_json": snp.detail_json,
        "latency_ms": snp.latency_ms,
        "created_at": snp.created_at,
    }


class MultimodalLoop:
    """定时截图 -> 多模态识别 -> 存证/告警。"""

    def __init__(self) -> None:
        self.config_svc = get_multimodal_config_service()
        self._stop = False
        self.last_run_at: float | None = None
        self.last_error: str | None = None
        self.stats = {"runs": 0, "alerts": 0, "errors": 0}

    async def run_forever(self) -> None:
        with SessionLocal() as db:
            self.config_svc.reload(db)
        cfg = self.config_svc.current()
        logger.info(f"多模态巡检循环已启动 enabled={cfg.enabled} interval={cfg.interval_seconds}s")
        while not self._stop:
            started = time.time()
            try:
                await self._tick()
            except Exception as exc:  # noqa: BLE001 顶层兜底，保证循环不退出
                self.stats["errors"] += 1
                self.last_error = str(exc)
                logger.exception(f"多模态巡检循环异常: {exc}")
            cfg = self.config_svc.current()
            wait = cfg.interval_seconds if cfg.enabled else 30
            await asyncio.sleep(max(1.0, wait - (time.time() - started)))

    def stop(self) -> None:
        self._stop = True

    async def _tick(self) -> None:
        with SessionLocal() as db:
            cfg = self.config_svc.reload(db)
            device_ids = [d.id for d in db.scalars(select(Device).where(Device.enabled == True)).all()]
        if not cfg.enabled:
            return

        for device_id in device_ids:
            try:
                res = await asyncio.to_thread(self._analyze_device_id, device_id, cfg)
            except MultimodalError as exc:
                self.stats["errors"] += 1
                self.last_error = str(exc)
                logger.warning(f"设备 {device_id} 多模态识别失败: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                self.stats["errors"] += 1
                self.last_error = str(exc)
                logger.exception(f"设备 {device_id} 多模态巡检失败: {exc}")
                continue
            self.stats["runs"] += 1
            if res["broadcast"]:
                self.stats["alerts"] += 1
                await manager.broadcast({"type": "alert", "data": res["broadcast"]})
        self.last_run_at = time.time()

    async def analyze_once(self, device_id: int) -> dict:
        """手动触发单设备识别，并广播告警（如有）。"""
        with SessionLocal() as db:
            cfg = self.config_svc.reload(db)
        res = await asyncio.to_thread(self._analyze_device_id, device_id, cfg)
        if res["broadcast"]:
            await manager.broadcast({"type": "alert", "data": res["broadcast"]})
        return res

    def _analyze_device_id(self, device_id: int, cfg: RuntimeConfig) -> dict:
        with SessionLocal() as db:
            device = db.get(Device, device_id)
            if device is None:
                raise ValueError(f"设备 {device_id} 不存在")
            packet = None
            try:
                packet = stream_service.get_frame(device)
            except Exception as exc:  # noqa: BLE001
                raise MultimodalError(f"取帧失败: {exc}") from exc
            frame = packet.frame

            result, raw_text, latency_ms = analyze_image(
                frame,
                provider=cfg.provider,
                model=cfg.model,
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.timeout_seconds,
                system_prompt=cfg.prompt_override or None,
            )

            snapshot_path = _save_snapshot(frame, device.id)
            snp = MonitoringSnapshot(
                device_id=device.id,
                resident_id=device.resident_id,
                provider=cfg.provider,
                model=cfg.model,
                event_type=result["event_type"],
                severity=result["severity"],
                confidence=result["confidence"],
                has_issue=result["has_issue"],
                level=AlertLevel(result["level"]),
                summary=result["summary"][:255],
                detail_json=json.dumps(
                    {
                        "details": result["details"],
                        "location_hint": result["location_hint"],
                        "suggestion": result["suggestion"],
                        "person_count": result["person_count"],
                        "raw": raw_text,
                    },
                    ensure_ascii=False,
                ),
                snapshot_path=snapshot_path,
                latency_ms=latency_ms,
            )
            db.add(snp)
            db.flush()

            broadcast = None
            if result["alert"]:
                broadcast = _emit_alert(db, device, result, snp.id, latency_ms, snapshot_path)
                snp.alert_id = broadcast["id"]

            db.commit()
            db.refresh(snp)
            snapshot_dict = _snapshot_dict(snp)
        return {"snapshot": snapshot_dict, "result": result, "broadcast": broadcast}

    def status(self) -> dict:
        cfg = self.config_svc.current()
        return {
            "running": not self._stop,
            "enabled": cfg.enabled,
            "provider": cfg.provider,
            "model": cfg.model,
            "interval_seconds": cfg.interval_seconds,
            "last_run_at": self.last_run_at,
            "last_error": self.last_error,
            "stats": self.stats,
        }


multimodal_loop = MultimodalLoop()
