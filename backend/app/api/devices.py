from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.device_manager import get_device_manager
from app.db import get_db
from app.models.entities import Device
from app.schemas.schemas import DeviceCreate, DeviceOut

router = APIRouter(prefix="/devices", tags=["设备管理"])


@router.get("", response_model=list[DeviceOut])
def list_devices(db: Session = Depends(get_db)):
    return db.scalars(select(Device).order_by(Device.id)).all()


@router.post("", response_model=DeviceOut)
def add_device(payload: DeviceCreate, db: Session = Depends(get_db)):
    exists = db.scalar(select(Device).where(Device.device_serial == payload.device_serial))
    if exists:
        raise HTTPException(status_code=409, detail="设备序列号已存在")
    device = Device(**payload.model_dump())
    db.add(device)
    db.commit()
    db.refresh(device)
    return device


@router.get("/{device_id}", response_model=DeviceOut)
def get_device(device_id: int, db: Session = Depends(get_db)):
    device = db.get(Device, device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    return device


@router.put("/{device_id}", response_model=DeviceOut)
def update_device(device_id: int, payload: DeviceCreate, db: Session = Depends(get_db)):
    device = db.get(Device, device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    for k, v in payload.model_dump().items():
        setattr(device, k, v)
    db.commit()
    db.refresh(device)
    return device


@router.delete("/{device_id}")
def delete_device(device_id: int, db: Session = Depends(get_db)):
    device = db.get(Device, device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    db.delete(device)
    db.commit()
    return {"ok": True}


@router.post("/sync")
def sync_devices(db: Session = Depends(get_db)):
    mgr = get_device_manager()
    added = mgr.sync_cloud_devices(db)
    return {"added": added}


@router.get("/{device_id}/live")
def live_address(device_id: int, db: Session = Depends(get_db)):
    device = db.get(Device, device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    mgr = get_device_manager()
    adapter = mgr.adapter(device.vendor)
    if adapter is None:
        return {"protocol": "sim", "url": None, "note": "模拟设备无真实取流地址"}
    addr = adapter.get_live_address({"device_serial": device.device_serial, "access_url": device.access_url, "channel_no": device.channel_no})
    return {"protocol": addr.protocol, "url": addr.url}