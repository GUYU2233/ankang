from datetime import datetime

import httpx

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.notify import manager
from app.core.secrets import decrypt_secret, encrypt_secret
from app.db import SessionLocal, get_db
from app.models.entities import AlertEvent, AlertFeedback, Device, FallEvent, Resident, WebhookConfig, MonitoringSnapshot
from app.services.ai_client import ai_client
from app.schemas.schemas import AlertActionIn, AlertAIVerifyOut, AlertOut, AlertVerifyIn, WebhookConfigCreate, WebhookConfigOut

router = APIRouter(prefix="/alerts", tags=["告警中心"])

# 处置状态机：动作 -> (允许的起始状态, 目标状态)
ALERT_TRANSITIONS = {
    "confirm": ("pending", "confirmed"),
    "handle": ("confirmed", "handled"),
    "close": ("handled", "closed"),
}

_STATUS_LABEL = {
    "pending": "待确认",
    "confirmed": "已确认",
    "handled": "已处置",
    "closed": "已归档",
}

_VALID_STATUS = {"pending", "confirmed", "handled", "closed"}

HUMAN_WEIGHT = 1.0
VISION_AI_WEIGHT = 0.35
DETECTOR_WEIGHT = 0.65


def _verify_target(alert: AlertEvent, requested: str | None = None) -> str:
    if requested in {"fall", "risk"}:
        return requested
    return "fall" if alert.event_type == "fall_event" else "risk"


def _detector_score(alert: AlertEvent, target: str) -> float:
    import json
    try:
        detail = json.loads(alert.detail_json or "{}")
    except (TypeError, json.JSONDecodeError):
        detail = {}
    key = "fall_prob" if target == "fall" else "score"
    return max(0.0, min(1.0, float(detail.get(key, detail.get("max_score", 0.0)) or 0.0)))


def _fuse_feedback(detector_score: float, ai: AlertFeedback | None, human: AlertFeedback | None) -> tuple[float, bool]:
    """人工有结果时拥有最终决定权；否则按 detector 0.65 + AI 0.35 融合。"""
    if human is not None:
        return (1.0 if human.decision else 0.0), bool(human.decision)
    values = [(detector_score, DETECTOR_WEIGHT)]
    if ai is not None:
        ai_prob = ai.confidence if ai.decision else 1.0 - ai.confidence
        values.append((ai_prob, VISION_AI_WEIGHT))
    score = sum(v * w for v, w in values) / sum(w for _, w in values)
    return round(score, 3), score >= 0.60


def apply_alert_action(db: Session, alert: AlertEvent, action: str, operator: str, note: str | None) -> AlertEvent:
    """按状态机执行动作；起始状态不匹配时抛 409。operator 为空则记为“值班员”。"""
    from_status, to_status = ALERT_TRANSITIONS[action]
    if alert.status != from_status:
        raise HTTPException(
            status_code=409,
            detail=f"当前状态为{_STATUS_LABEL.get(alert.status, alert.status)}，不允许该操作",
        )
    operator = operator or "值班员"
    alert.status = to_status
    if action == "confirm":
        alert.confirmed = True
        alert.confirmed_by = operator
        alert.confirmed_at = datetime.now()
        alert.confirm_note = note
    elif action == "handle":
        alert.handled = True
        alert.handled_by = operator
        alert.handled_at = datetime.now()
        alert.handle_note = note
    elif action == "close":
        alert.closed_at = datetime.now()
    db.commit()
    db.refresh(alert)
    return alert


def enrich_alerts(db: Session, alerts: list[AlertEvent]) -> list[AlertOut]:
    """按 resident_id 批量富化老人姓名与监护人电话（dashboard.py 复用）。"""
    resident_ids = {a.resident_id for a in alerts if a.resident_id}
    residents: dict[int, Resident] = {}
    if resident_ids:
        rows = db.scalars(select(Resident).where(Resident.id.in_(resident_ids))).all()
        residents = {r.id: r for r in rows}
    out: list[AlertOut] = []
    for a in alerts:
        obj = AlertOut.model_validate(a)
        r = residents.get(a.resident_id)
        if r:
            obj.resident_name = r.name
            obj.guardian_phone = r.guardian_phone or ""
        out.append(obj)
    return out


@router.get("", response_model=list[AlertOut])
def list_alerts(
    level: str | None = None,
    status: str | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    if status is not None and status not in _VALID_STATUS:
        raise HTTPException(status_code=400, detail="非法的状态筛选值")
    stmt = select(AlertEvent)
    if level:
        stmt = stmt.where(AlertEvent.level == level)
    if status:
        stmt = stmt.where(AlertEvent.status == status)
    stmt = stmt.order_by(AlertEvent.created_at.desc()).limit(min(limit, 500))
    return enrich_alerts(db, db.scalars(stmt).all())


# ========== Webhook 配置管理 ==========

@router.get("/webhooks", response_model=list[WebhookConfigOut])
def list_webhooks(db: Session = Depends(get_db)):
    rows = db.scalars(select(WebhookConfig).order_by(WebhookConfig.created_at.desc())).all()
    out = []
    for r in rows:
        obj = WebhookConfigOut.model_validate(r)
        url = r.webhook_url or ""
        obj.webhook_url = ""
        obj.webhook_url_masked = url[:30] + "..." if len(url) > 30 else url
        obj.secret_present = bool(r.secret)
        out.append(obj)
    return out


@router.post("/webhooks", response_model=WebhookConfigOut)
def create_webhook(payload: WebhookConfigCreate, db: Session = Depends(get_db)):
    cfg = WebhookConfig(
        name=payload.name,
        platform=payload.platform,
        webhook_url=payload.webhook_url,
        secret=encrypt_secret(payload.secret, "webhook.secret") if payload.secret else None,
        enabled=payload.enabled,
        trigger_levels=payload.trigger_levels,
    )
    db.add(cfg)
    db.commit()
    db.refresh(cfg)
    obj = WebhookConfigOut.model_validate(cfg)
    url = cfg.webhook_url or ""
    obj.webhook_url = ""
    obj.webhook_url_masked = url[:30] + "..." if len(url) > 30 else url
    obj.secret_present = bool(cfg.secret)
    return obj


@router.put("/webhooks/{hook_id}", response_model=WebhookConfigOut)
def update_webhook(hook_id: int, payload: WebhookConfigCreate, db: Session = Depends(get_db)):
    cfg = db.get(WebhookConfig, hook_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail="Webhook 配置不存在")
    for k, v in payload.model_dump(exclude_unset=True).items():
        if k == "secret":
            if not v:
                continue
            v = encrypt_secret(v, "webhook.secret")
        setattr(cfg, k, v)
    db.commit()
    db.refresh(cfg)
    obj = WebhookConfigOut.model_validate(cfg)
    url = cfg.webhook_url or ""
    obj.webhook_url = ""
    obj.webhook_url_masked = url[:30] + "..." if len(url) > 30 else url
    obj.secret_present = bool(cfg.secret)
    return obj


@router.delete("/webhooks/{hook_id}")
def delete_webhook(hook_id: int, db: Session = Depends(get_db)):
    cfg = db.get(WebhookConfig, hook_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail="Webhook 配置不存在")
    db.delete(cfg)
    db.commit()
    return {"ok": True}


@router.post("/webhooks/{hook_id}/test")
async def test_webhook(hook_id: int, db: Session = Depends(get_db)):
    """发送测试消息到指定 webhook。"""
    from app.core.notify import send_webhook
    cfg = db.get(WebhookConfig, hook_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail="Webhook 配置不存在")
    test_alert = {
        "title": "测试消息 - 智护安康",
        "level": "red",
        "device_name": "测试设备",
        "scene": "测试",
        "resident_name": "测试老人",
        "score": 0.95,
        "fall_prob": 0.92,
        "created_at": datetime.now().isoformat(),
    }
    ok = await send_webhook(cfg.webhook_url, cfg.platform, decrypt_secret(cfg.secret, "webhook.secret"), test_alert)
    return {"ok": ok, "detail": "发送成功" if ok else "发送失败，请检查 webhook 地址和网络"}

@router.get("/{alert_id}/replay")
def alert_replay_status(alert_id: int, db: Session = Depends(get_db)):
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    if not alert.replay_clip_id:
        return {"status": "unavailable", "progress": 0, "message": "该告警没有关联回放"}
    status = ai_client.recording_status(alert.replay_clip_id)
    if status is None:
        return {"status": "failed", "progress": 0, "message": "回放服务暂不可用"}
    now = datetime.now().timestamp()
    start = float(status.get("started_at", now)); end = float(status.get("end_at", now))
    progress = 100 if status.get("status") == "ready" else max(0, min(99, int((now - start) / max(1, end - start) * 100)))
    return {"status": status.get("status"), "progress": progress, "clip_id": alert.replay_clip_id, "frame_count": status.get("frame_count", 0), "message": status.get("error", "")}


@router.get("/{alert_id}/replay/video.mp4")
async def alert_replay_video(alert_id: int, request: Request, db: Session = Depends(get_db)):
    alert = db.get(AlertEvent, alert_id)
    if alert is None or not alert.replay_clip_id:
        raise HTTPException(status_code=404, detail="告警回放不存在")
    url = ai_client.recording_video_url(alert.replay_clip_id)
    upstream_headers = {"Range": request.headers["range"]} if request.headers.get("range") else {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=upstream_headers)
    if resp.status_code not in {200, 206}:
        raise HTTPException(status_code=404, detail="回放尚未生成")
    response_headers = {"Cache-Control": "private, max-age=3600", "Accept-Ranges": "bytes"}
    for name in ("content-range", "content-length"):
        if name in resp.headers:
            response_headers[name.title()] = resp.headers[name]
    return Response(content=resp.content, status_code=resp.status_code, media_type="video/mp4", headers=response_headers)


@router.get("/{alert_id}/image")
def alert_image(alert_id: int, db: Session = Depends(get_db)):
    """返回告警证据帧；仅允许读取受控 snapshot 目录。"""
    import json
    from pathlib import Path
    from fastapi.responses import FileResponse, StreamingResponse
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    try:
        path_value = json.loads(alert.detail_json or "{}").get("snapshot_path") or ""
    except (TypeError, json.JSONDecodeError):
        path_value = ""
    path = Path(path_value).resolve() if path_value else None
    root = (Path(__file__).resolve().parents[3] / "data" / "snapshots").resolve()
    if path is None or root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="证据截图不存在")
    return FileResponse(str(path), media_type="image/jpeg")


@router.get("/{alert_id}", response_model=AlertOut)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    return enrich_alerts(db, [alert])[0]


@router.post("/{alert_id}/verify", response_model=AlertOut)
def verify_alert(alert_id: int, payload: AlertVerifyIn, db: Session = Depends(get_db)):
    """人工现场核验。人工结论拥有最终权重，并产出可审计训练反馈。"""
    import json
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    if alert.status != "pending":
        raise HTTPException(status_code=409, detail="该告警已经核验")
    target = _verify_target(alert, payload.target)
    expected = "fall" if alert.event_type == "fall_event" else "risk"
    if target != expected:
        raise HTTPException(status_code=422, detail="核验目标与告警类型不一致")
    detail = json.loads(alert.detail_json or "{}")
    detector_score = _detector_score(alert, target)
    detector_feedback = AlertFeedback(
        alert_id=alert.id, device_id=alert.device_id, source="detector", target=target,
        decision=detector_score >= 0.60, confidence=detector_score, weight=DETECTOR_WEIGHT,
        snapshot_path=detail.get("snapshot_path") or None,
        metadata_json=json.dumps({"event_type": alert.event_type}, ensure_ascii=False),
    )
    db.add(detector_feedback)
    feedback = AlertFeedback(
        alert_id=alert.id, device_id=alert.device_id, source="human", target=target,
        decision=payload.decision, confidence=1.0, weight=HUMAN_WEIGHT,
        operator=payload.operator or "值班员", note=payload.note,
        snapshot_path=detail.get("snapshot_path") or None,
        metadata_json=json.dumps({"event_type": alert.event_type, "detector_score": _detector_score(alert, target)}, ensure_ascii=False),
    )
    db.add(feedback)
    answer = "确认发生跌倒" if target == "fall" and payload.decision else "确认未发生跌倒" if target == "fall" else "确认存在跌倒风险" if payload.decision else "确认暂未发现跌倒风险"
    apply_alert_action(db, alert, "confirm", payload.operator, f"{answer}" + (f"；{payload.note}" if payload.note else ""))
    if not payload.decision:
        alert.handled = True
        alert.status = "handled"
        alert.handled_by = payload.operator or "值班员"
        alert.handled_at = datetime.now()
        alert.handle_note = "现场核验为误报/无风险，已自动处置"
        active = db.scalar(select(FallEvent).where(FallEvent.alert_id == alert.id, FallEvent.end_at.is_(None)))
        if active:
            active.end_at = datetime.now()
    db.commit(); db.refresh(alert)
    return enrich_alerts(db, [alert])[0]


@router.post("/{alert_id}/ai-verify", response_model=AlertAIVerifyOut)
async def ai_verify_alert(alert_id: int, db: Session = Depends(get_db)):
    """调用视觉巡检模型做低权重二次确认，不覆盖人工结论。"""
    import json
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    device = db.get(Device, alert.device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="来源设备不存在")
    from app.services.multimodal_loop import multimodal_loop
    try:
        # 辅助核验只生成快照与判断，不在此处再产生第二条告警。
        with SessionLocal() as cfg_db:
            cfg = multimodal_loop.config_svc.reload(cfg_db)
        analyzed = await __import__('asyncio').to_thread(multimodal_loop._analyze_device_id, device.id, cfg, False)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI 风险确认失败: {exc}") from exc
    result = analyzed["result"]
    target = _verify_target(alert)
    decision = result.get("event_type") == "fall" if target == "fall" else bool(result.get("has_issue"))
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    detector_score = _detector_score(alert, target)
    detector_existing = db.scalar(select(AlertFeedback).where(AlertFeedback.alert_id == alert.id, AlertFeedback.source == "detector").order_by(AlertFeedback.created_at.desc()))
    if detector_existing is None:
        db.add(AlertFeedback(alert_id=alert.id, device_id=device.id, source="detector", target=target, decision=detector_score >= 0.60, confidence=detector_score, weight=DETECTOR_WEIGHT))
    feedback = AlertFeedback(
        alert_id=alert.id, device_id=device.id, source="vision_ai", target=target,
        decision=decision, confidence=confidence, weight=VISION_AI_WEIGHT,
        note=result.get("summary") or "", snapshot_path=analyzed["snapshot"].get("snapshot_path"),
        metadata_json=json.dumps({"provider": analyzed["snapshot"].get("provider"), "model": analyzed["snapshot"].get("model"), "suggestion": result.get("suggestion", "")}, ensure_ascii=False),
    )
    db.add(feedback); db.commit()
    human = db.scalar(select(AlertFeedback).where(AlertFeedback.alert_id == alert.id, AlertFeedback.source == "human").order_by(AlertFeedback.created_at.desc()))
    fused_score, fused_decision = _fuse_feedback(_detector_score(alert, target), feedback, human)
    return AlertAIVerifyOut(alert_id=alert.id, target=target, decision=decision, confidence=confidence, weight=VISION_AI_WEIGHT, summary=result.get("summary", ""), suggestion=result.get("suggestion", ""), fused_score=fused_score, fused_decision=fused_decision)


@router.get("/{alert_id}/feedback")
def alert_feedback(alert_id: int, db: Session = Depends(get_db)):
    rows = db.scalars(select(AlertFeedback).where(AlertFeedback.alert_id == alert_id).order_by(AlertFeedback.created_at)).all()
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    target = _verify_target(alert)
    ai = next((x for x in reversed(rows) if x.source == "vision_ai"), None)
    human = next((x for x in reversed(rows) if x.source == "human"), None)
    score, decision = _fuse_feedback(_detector_score(alert, target), ai, human)
    return {"alert_id": alert_id, "target": target, "fused_score": score, "fused_decision": decision, "human_overrides": human is not None, "weights": {"detector": DETECTOR_WEIGHT, "vision_ai": VISION_AI_WEIGHT, "human": HUMAN_WEIGHT}, "feedback": [{"source": x.source, "decision": x.decision, "confidence": x.confidence, "weight": x.weight, "note": x.note, "created_at": x.created_at} for x in rows]}


@router.post("/{alert_id}/confirm", response_model=AlertOut)
def confirm_alert(alert_id: int, payload: AlertActionIn, db: Session = Depends(get_db)):
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    apply_alert_action(db, alert, "confirm", payload.operator, payload.note)
    return enrich_alerts(db, [alert])[0]


@router.post("/{alert_id}/handle", response_model=AlertOut)
def handle_alert(alert_id: int, payload: AlertActionIn, db: Session = Depends(get_db)):
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    apply_alert_action(db, alert, "handle", payload.operator, payload.note)
    return enrich_alerts(db, [alert])[0]


@router.post("/{alert_id}/close", response_model=AlertOut)
def close_alert(alert_id: int, payload: AlertActionIn, db: Session = Depends(get_db)):
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    apply_alert_action(db, alert, "close", payload.operator, payload.note)
    return enrich_alerts(db, [alert])[0]


@router.post("/{alert_id}/ack")
def ack_alert(alert_id: int, handled: bool = True, db: Session = Depends(get_db)):
    """旧接口兼容：置 confirmed=True，handled 取参数；status 按当前状态映射。"""
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    alert.confirmed = True
    alert.handled = handled
    if alert.status != "closed":
        alert.status = "handled" if handled else "confirmed"
    db.commit()
    return {"ok": True}


@router.get("/stats/today")
def today_stats(db: Session = Depends(get_db)):
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    alert_count = len(db.scalars(select(AlertEvent).where(AlertEvent.created_at >= start)).all())
    fall_count = len(db.scalars(select(FallEvent).where(FallEvent.start_at >= start)).all())
    return {"today_alerts": alert_count, "today_falls": fall_count}
