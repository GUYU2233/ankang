from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from loguru import logger
from sqlalchemy import select
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
        self._recovery_counters: dict[int, int] = defaultdict(int)
        self.emit_cooldown_seconds = 15
        self.aggregate_window_seconds = 30
        self.recovery_confirm_frames = 3
        self.snapshot_dir = Path(__file__).resolve().parents[3] / "data" / "snapshots" / "pose"

    def process(self, db: Session, device: Device, infer, current_score: float, frame_at: float, evidence_frame=None):
        """处理一帧推理结果；evidence_frame 可选，告警确认时保存带标注证据帧。"""
        resident_id = device.resident_id
        subject_key = f"r{resident_id or 0}:d{device.id}"

        result = self.risk_engine.score(subject_key, device.scene, infer, frame_at)
        score = result.score
        fall = bool(infer.fall_detected)
        level: AlertLevel = result.level
        self._reconcile_fall_event(db, device, fall, getattr(infer, "fall_prob", 0.0))

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
            snapshot_path = self._save_evidence(evidence_frame, device.id) if fall and evidence_frame is not None else None
            broadcast = self._emit(db, device, resident_id, level, score, infer, result.factors, result.events, result.trigger_reason, snapshot_path)
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
        snapshot_path: str | None = None,
    ) -> dict | None:
        now = datetime.now()
        event_type = "fall_event" if infer.fall_detected else ("fall_risk" if score >= 0.55 else "behavior_risk")

        # 同设备、同事件类型在短窗口内聚合，避免连续帧重复创建告警。
        recent = db.scalar(
            select(AlertEvent).where(
                AlertEvent.device_id == device.id,
                AlertEvent.event_type == event_type,
                AlertEvent.created_at >= now - timedelta(seconds=self.aggregate_window_seconds),
                AlertEvent.status != "closed",
            ).order_by(AlertEvent.created_at.desc())
        )
        if recent is not None:
            try:
                detail = json.loads(recent.detail_json or "{}")
            except (TypeError, json.JSONDecodeError):
                detail = {}
            detail["occurrence_count"] = int(detail.get("occurrence_count", 1)) + 1
            detail["last_seen_at"] = now.isoformat()
            detail["max_score"] = max(float(detail.get("max_score", detail.get("score", 0.0))), float(score))
            detail["max_fall_prob"] = max(float(detail.get("max_fall_prob", detail.get("fall_prob", 0.0))), float(getattr(infer, "fall_prob", 0.0)))
            if snapshot_path and not detail.get("snapshot_path"):
                detail["snapshot_path"] = snapshot_path
            recent.detail_json = json.dumps(detail, ensure_ascii=False)
            if level == AlertLevel.RED and recent.level != AlertLevel.RED:
                recent.level = AlertLevel.RED
                recent.title = f"{device.scene}检测到疑似跌倒"
            return None
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
                "occurrence_count": 1,
                "last_seen_at": now.isoformat(),
                "max_score": score,
                "max_fall_prob": getattr(infer, "fall_prob", 0.0),
                "snapshot_path": snapshot_path or "",
            }, ensure_ascii=False),
            confirmed=False,
            handled=False,
            created_at=now,
        )
        db.add(alert)
        db.flush()

        db.add(NotificationLog(alert_id=alert.id, channel="websocket", target="家属端", content=title, status="sent"))

        resident = db.get(Resident, resident_id) if resident_id else None

        # 异步分发 webhook 通知（微信/钉钉/飞书等）。resident 必须先加载。
        self._dispatch_webhooks(db, alert, device, resident, level, score, infer)

        if infer.fall_detected:
            active_fall = db.scalar(
                select(FallEvent).where(FallEvent.device_id == device.id, FallEvent.end_at.is_(None))
                .order_by(FallEvent.start_at.desc())
            )
            if active_fall is None:
                active_fall = FallEvent(device_id=device.id, resident_id=resident_id, start_at=now)
                db.add(active_fall)
            active_fall.alert_id = active_fall.alert_id or alert.id
            active_fall.fall_prob = max(float(active_fall.fall_prob or 0.0), float(infer.fall_prob))
            active_fall.screenshot_path = active_fall.screenshot_path or snapshot_path
            active_fall.note = infer.fall_type

        resident = db.get(Resident, resident_id) if resident_id else None
        guardian_phone = (resident.guardian_phone or "") if resident else ""
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
            "guardian_phone": guardian_phone,
            "score": score,
            "fall_prob": infer.fall_prob,
            "nearfall_prob": getattr(infer, "nearfall_prob", 0.0),
            "trigger_reason": trigger_reason,
            "status": "pending",
            "created_at": now.isoformat(),
        }

    def _save_evidence(self, frame, device_id: int) -> str | None:
        """保存姿态告警证据帧；写盘失败只记日志，不阻断告警。"""
        if frame is None:
            return None
        try:
            day_dir = self.snapshot_dir / datetime.now().strftime("%Y%m%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            path = day_dir / f"dev{device_id}_{uuid.uuid4().hex[:12]}.jpg"
            # Windows OpenCV 对非 ASCII 路径的 imwrite 支持不稳定，使用 imencode + Python 写盘。
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                logger.warning(f"告警证据帧编码失败: {path.name}")
                return None
            path.write_bytes(buf.tobytes())
            return str(path.resolve())
        except Exception as exc:
            logger.warning(f"告警证据帧保存异常: {exc}")
            return None

    def _reconcile_fall_event(self, db: Session, device: Device, fall: bool, fall_prob: float) -> None:
        """维护每设备唯一活动跌倒事件，并在连续非跌倒帧后写入恢复时间。"""
        active = db.scalar(
            select(FallEvent).where(FallEvent.device_id == device.id, FallEvent.end_at.is_(None))
            .order_by(FallEvent.start_at.desc())
        )
        if fall:
            self._recovery_counters[device.id] = 0
            if active is not None:
                active.fall_prob = max(float(active.fall_prob or 0.0), float(fall_prob or 0.0))
            return
        if active is None:
            self._recovery_counters[device.id] = 0
            return
        self._recovery_counters[device.id] += 1
        if self._recovery_counters[device.id] >= self.recovery_confirm_frames:
            active.end_at = datetime.now()
            self._recovery_counters[device.id] = 0
            logger.info(f"跌倒事件恢复: device={device.id} fall_event={active.id}")

    def _dispatch_webhooks(self, db: Session, alert: AlertEvent, device: Device, resident, level, score, infer) -> None:
        """异步分发告警到已启用的 webhook 渠道。"""
        from app.core.notify import send_webhook
        from app.models.entities import WebhookConfig
        from sqlalchemy import select

        rows = db.scalars(select(WebhookConfig).where(WebhookConfig.enabled == True)).all()
        if not rows:
            return
        alert_data = {
            "title": alert.title,
            "level": level.value,
            "device_name": device.device_name,
            "scene": device.scene,
            "resident_name": resident.name if resident else "",
            "score": round(score, 2),
            "fall_prob": getattr(infer, "fall_prob", 0.0),
            "created_at": alert.created_at.isoformat() if alert.created_at else "",
        }
        for cfg in rows:
            trigger_levels = [l.strip() for l in (cfg.trigger_levels or "red,orange").split(",") if l.strip()]
            if level.value not in trigger_levels:
                continue
            try:
                import asyncio
                from app.core.secrets import decrypt_secret
                secret = decrypt_secret(cfg.secret, "webhook.secret")
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None:
                    loop.create_task(send_webhook(cfg.webhook_url, cfg.platform, secret, alert_data))
                else:
                    logger.info(f"Webhook [{cfg.name}] 将由异步检测循环分发；当前为同步上下文")
            except Exception as exc:
                logger.warning(f"Webhook dispatch error [{cfg.name}]: {exc}")
