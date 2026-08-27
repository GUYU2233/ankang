from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.stream_service import stream_service
from app.db import get_db
from app.models.entities import Device

router = APIRouter(prefix="/streams", tags=["视频流"])


@router.get("/{device_id}/frame.jpg")
def get_frame_jpeg(device_id: int, db: Session = Depends(get_db)):
    device = db.get(Device, device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    try:
        packet = stream_service.get_frame(device)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"取流失败: {exc}")
    jpeg = stream_service.encode_jpeg(packet.frame)
    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/{device_id}/meta")
def get_frame_meta(device_id: int, db: Session = Depends(get_db)):
    device = db.get(Device, device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    packet = stream_service.get_frame(device)
    return {"device_id": device_id, "meta": packet.meta, "captured_at": packet.captured_at}