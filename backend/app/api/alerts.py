from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.notify import manager
from app.db import get_db
from app.models.entities import AlertEvent, FallEvent
from app.schemas.schemas import AlertOut

router = APIRouter(prefix="/alerts", tags=["告警中心"])


@router.get("", response_model=list[AlertOut])
def list_alerts(level: str | None = None, limit: int = 100, db: Session = Depends(get_db)):
    stmt = select(AlertEvent).order_by(AlertEvent.created_at.desc()).limit(min(limit, 500))
    if level:
        stmt = select(AlertEvent).where(AlertEvent.level == level).order_by(AlertEvent.created_at.desc()).limit(min(limit, 500))
    return db.scalars(stmt).all()


@router.get("/{alert_id}", response_model=AlertOut)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    return alert


@router.post("/{alert_id}/ack")
def ack_alert(alert_id: int, handled: bool = True, db: Session = Depends(get_db)):
    alert = db.get(AlertEvent, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    alert.confirmed = True
    alert.handled = handled
    db.commit()
    return {"ok": True}


@router.get("/stats/today")
def today_stats(db: Session = Depends(get_db)):
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    alert_count = len(db.scalars(select(AlertEvent).where(AlertEvent.created_at >= start)).all())
    fall_count = len(db.scalars(select(FallEvent).where(FallEvent.start_at >= start)).all())
    return {"today_alerts": alert_count, "today_falls": fall_count}