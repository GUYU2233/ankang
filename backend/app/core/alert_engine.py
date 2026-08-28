from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime

from loguru import logger
from sqlalchemy.orm import Session

from app.core.risk_engine import RiskEngine
from app.models.entities import AlertEvent, AlertLevel, Device, FallEvent, NotificationLog, Resident, RiskScore


class AlertEngine:
    """分级预警引擎。实现绿/黄/橙/红四级，含连续帧确认的误报抑制与事件归档。"""

    def __init__(self, risk_engine: RiskEngine | None = None, confirm_frames: int = 2) -> None:
        self.risk_engine = risk_engine or RiskEngine()
        self.confirm_frames = confirm_frames
        self._counters: dict[int, int] = defaultdict(int)
        self._last_emit_ts: dict[int, float] = defaultdict(float)
        self.emit_cooldown_seconds = 15

    def process(self, db: Session, device: Device, infer, current_score: float, frame_at: float):
        """处理一帧推理结果，返回是否需要广播的告警字典。"""
        resident_id = device.resident_id
        subject_key = f"r{resident_id or 0}:d{device.id}"

        result = self.risk_engine.score(subject_key, device.scene, infer, frame_at)
        score = result.score
        fall = bool(infer.fall_detected)
        level: AlertLevel = result.level

        # 记录风险评分（风险档案）
        if score >= 0.20 or fall:
            db.add(RiskScore(
                resident_id=resident_id,
                device_id=device.id,
                score=score,
                level=level,
                factors_json=self.risk_engine.factors_json(result.factors),
                created_at=datetime.now(),
            ))

        broadcast = None
        if level.value in ("red", "orange", "yellow"):
            self._counters[device.id] += 1
        else:
            self._counters[device.id] = 0

        should_emit = False
        if fall:
            should_emit = self._counters[device.id] >= self.confirm_frames
        elif level.value in ("orange", "yellow"):
            should_emit = self._counters[device.id] >= self.confirm_frames

        if should_emit and (frame_at - self._last_emit_ts.get(device.id, 0)) >= self.emit_cooldown_seconds:
            self._last_emit_ts[device.id] = frame_at
            broadcast = self._emit(db, device, resident_id, level, score, infer, result.factors, result.events, result.trigger_reason)
            self._counters[device.id] = 0

        db.commit()
        return broadcast

    def _emit(
        self,
        db: Session,
        device: Device,
        resident_id: int | None,
        level: AlertLevel,
        score: float,
        infer,
        factors: list[dict],
        events: list[str],
        trigger_reason: str = "",
    ) -> dict:
        now = datetime.now()
        event_type = "fall_event" if infer.fall_detected else ("fall_risk" if score >= 0.55 else "behavior_risk")
        if infer.fall_detected:
            title = f"{device.scene}检测到疑似跌倒"
        elif level == AlertLevel.ORANGE:
            title = f"{device.scene}跌倒高风险预警"
        elif level == AlertLevel.YELLOW:
            title = f"{device.scene}跌倒风险关注提醒"
        else:
            title = f"{device.scene}行为风险提示"

        alert = AlertEvent(
            alert_no=uuid.uuid4().hex[:16],
            device_id=device.id,
            resident_id=resident_id,
            level=level,
            event_type=event_type,
            title=title,
            detail_json=json.dumps({
                "score": score,
                "fall_prob": infer.fall_prob,
                "nearfall_prob": getattr(infer, "nearfall_prob", 0.0),
                "gait_unsteadiness": getattr(infer, "gait_unsteadiness", 0.0),
                "fall_type": infer.fall_type,
                "factors": factors,
                "events": events,
                "trigger_reason": trigger_reason,
                "mock": infer.mock,
            }, ensure_ascii=False),
            confirmed=False,
            handled=False,
            created_at=now,
        )
        db.add(alert)
        db.flush()

        db.add(NotificationLog(alert_id=alert.id, channel="websocket", target="家属端", content=title, status="sent"))

        if infer.fall_detected:
            db.add(FallEvent(
                device_id=device.id,
                resident_id=resident_id,
                alert_id=alert.id,
                start_at=now,
                fall_prob=infer.fall_prob,
                note=infer.fall_type,
            ))

        resident = db.get(Resident, resident_id) if resident_id else None
        logger.info(f"告警触发: {title} level={level.value} score={score}")
        return {
            "id": alert.id,
            "alert_no": alert.alert_no,
            "level": level.value,
            "event_type": event_type,
            "title": title,
            "device_id": device.id,
            "device_name": device.device_name,
            "scene": device.scene,
            "resident_id": resident_id,
            "resident_name": resident.name if resident else "",
            "score": score,
            "fall_prob": infer.fall_prob,
            "nearfall_prob": getattr(infer, "nearfall_prob", 0.0),
            "trigger_reason": trigger_reason,
            "created_at": now.isoformat(),
        }
