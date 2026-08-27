from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.entities import AlertEvent, Device, Resident, RiskScore
from app.schemas.schemas import AlertOut, DashboardStats

router = APIRouter(prefix="/dashboard", tags=["数据看板"])


@router.get("/stats", response_model=DashboardStats)
def stats(db: Session = Depends(get_db)):
    devices = db.scalars(select(Device)).all()
    residents = db.scalars(select(Resident)).all()
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    alerts = db.scalars(select(AlertEvent).where(AlertEvent.created_at >= start).order_by(AlertEvent.created_at.desc()).limit(10)).all()
    latest_score = db.scalar(select(RiskScore).order_by(RiskScore.created_at.desc()))
    return DashboardStats(
        total_devices=len(devices),
        online_devices=sum(1 for d in devices if d.status == "online"),
        total_residents=len(residents),
        today_alerts=len(db.scalars(select(AlertEvent).where(AlertEvent.created_at >= start)).all()),
        today_falls=len(db.scalars(select(AlertEvent).where(AlertEvent.created_at >= start, AlertEvent.event_type == "fall_event")).all()),
        avg_risk_score=round(latest_score.score, 2) if latest_score else 0.0,
        latest_alerts=[AlertOut.model_validate(a) for a in alerts],
    )