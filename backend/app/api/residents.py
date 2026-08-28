import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.dashboard import aggregate_risk_trend
from app.api.devices import enrich_devices
from app.db import get_db
from app.models.entities import Device, Resident, RiskScore
from app.schemas.schemas import DeviceOut, ResidentCreate, ResidentOut, ResidentRiskProfileOut, RiskFactor

router = APIRouter(prefix="/residents", tags=["老人档案"])


@router.get("", response_model=list[ResidentOut])
def list_residents(db: Session = Depends(get_db)):
    return db.scalars(select(Resident).order_by(Resident.id)).all()


@router.post("", response_model=ResidentOut)
def add_resident(payload: ResidentCreate, db: Session = Depends(get_db)):
    resident = Resident(**payload.model_dump())
    db.add(resident)
    db.commit()
    db.refresh(resident)
    return resident


@router.get("/{resident_id}/devices", response_model=list[DeviceOut])
def resident_devices(resident_id: int, db: Session = Depends(get_db)):
    resident = db.get(Resident, resident_id)
    if resident is None:
        raise HTTPException(status_code=404, detail="老人档案不存在")
    devices = db.scalars(select(Device).where(Device.resident_id == resident_id).order_by(Device.id)).all()
    return enrich_devices(db, devices)


@router.get("/{resident_id}/risk-profile", response_model=ResidentRiskProfileOut)
def resident_risk_profile(resident_id: int, days: int = Query(7, ge=1, le=90), db: Session = Depends(get_db)):
    resident = db.get(Resident, resident_id)
    if resident is None:
        raise HTTPException(status_code=404, detail="老人档案不存在")
    latest = db.scalar(
        select(RiskScore).where(RiskScore.resident_id == resident_id).order_by(RiskScore.created_at.desc())
    )
    factors: list[RiskFactor] = []
    if latest and latest.factors_json:
        try:
            raw = json.loads(latest.factors_json)
            for f in raw:
                if isinstance(f, dict):
                    factors.append(RiskFactor(
                        key=f.get("key", ""),
                        label=f.get("label", ""),
                        value=float(f.get("value", 0.0) or 0.0),
                        unit=f.get("unit", "") or "",
                        normal_range=f.get("normal_range", "") or "",
                    ))
        except Exception:
            factors = []
    trend = aggregate_risk_trend(db, days, resident_id=resident_id)
    return ResidentRiskProfileOut(
        resident_id=resident_id,
        resident_name=resident.name,
        latest_score=latest.score if latest else 0.0,
        latest_level=latest.level.value if latest else "green",
        factors=factors,
        trend=trend,
        updated_at=latest.created_at if latest else None,
    )


@router.put("/{resident_id}", response_model=ResidentOut)
def update_resident(resident_id: int, payload: ResidentCreate, db: Session = Depends(get_db)):
    resident = db.get(Resident, resident_id)
    if resident is None:
        raise HTTPException(status_code=404, detail="老人档案不存在")
    for k, v in payload.model_dump().items():
        setattr(resident, k, v)
    db.commit()
    db.refresh(resident)
    return resident


@router.delete("/{resident_id}")
def delete_resident(resident_id: int, db: Session = Depends(get_db)):
    resident = db.get(Resident, resident_id)
    if resident is None:
        raise HTTPException(status_code=404, detail="老人档案不存在")
    db.delete(resident)
    db.commit()
    return {"ok": True}
