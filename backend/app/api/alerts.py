from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.notify import manager
from app.db import get_db
from app.models.entities import AlertEvent, FallEvent, Resident
from app.schemas.schemas import AlertActionIn, AlertOut

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


@router.get("/{alert_id}", response_model=AlertOut)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    return enrich_alerts(db, [alert])[0]


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
