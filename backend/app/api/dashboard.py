from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.alerts import enrich_alerts
from app.db import get_db
from app.models.entities import AlertEvent, Device, Resident, RiskScore
from app.schemas.schemas import DashboardStats, RiskTrendPoint

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
        latest_alerts=enrich_alerts(db, alerts),
    )


def aggregate_risk_trend(db: Session, days: int, resident_id: int | None = None) -> list[RiskTrendPoint]:
    """按日聚合风险评分，生成含今天在内的连续 days 个日期（升序），无数据日补 0。"""
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
    stmt = select(
        func.date(RiskScore.created_at).label("d"),
        func.avg(RiskScore.score).label("avg_score"),
        func.max(RiskScore.score).label("max_score"),
        func.count(RiskScore.id).label("cnt"),
    ).where(RiskScore.created_at >= start)
    if resident_id is not None:
        stmt = stmt.where(RiskScore.resident_id == resident_id)
    stmt = stmt.group_by(func.date(RiskScore.created_at))
    by_date: dict[str, tuple[float, float, int]] = {}
    for row in db.execute(stmt):
        by_date[row.d] = (round(float(row.avg_score), 3), round(float(row.max_score), 3), int(row.cnt))
    result: list[RiskTrendPoint] = []
    for i in range(days):
        ds = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        if ds in by_date:
            avg_score, max_score, cnt = by_date[ds]
            result.append(RiskTrendPoint(date=ds, avg_score=avg_score, max_score=max_score, count=cnt))
        else:
            result.append(RiskTrendPoint(date=ds))
    return result


@router.get("/risk-trend", response_model=list[RiskTrendPoint])
def risk_trend(days: int = Query(7, ge=1, le=90), db: Session = Depends(get_db)):
    return aggregate_risk_trend(db, days)
