from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.entities import Resident
from app.schemas.schemas import ResidentCreate, ResidentOut

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