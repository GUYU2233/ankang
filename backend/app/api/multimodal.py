"""多模态视觉巡检 REST 接口。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.entities import MonitoringSnapshot, MultimodalConfigRecord
from app.schemas.schemas import (
    AnalyzeResponse,
    MultimodalConfigOut,
    MultimodalConfigUpdate,
    ProviderPreset,
    SnapshotOut,
)
from app.services.multimodal_client import PROVIDER_PRESETS, MultimodalError
from app.services.multimodal_config import get_multimodal_config_service
from app.services.multimodal_loop import multimodal_loop

router = APIRouter(prefix="/multimodal", tags=["多模态巡检"])

config_svc = get_multimodal_config_service()


def _config_out(db: Session) -> MultimodalConfigOut:
    cfg = config_svc.reload(db)
    rec = db.get(MultimodalConfigRecord, 1)
    return MultimodalConfigOut(
        provider=cfg.provider,
        model=cfg.model,
        base_url=cfg.base_url,
        api_key_masked=cfg.mask_key(),
        enabled=cfg.enabled,
        interval_seconds=cfg.interval_seconds,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        prompt_override=cfg.prompt_override,
        updated_at=rec.updated_at if rec else None,
    )


@router.get("/config", response_model=MultimodalConfigOut)
def get_config(db: Session = Depends(get_db)):
    return _config_out(db)


@router.put("/config", response_model=MultimodalConfigOut)
def update_config(payload: MultimodalConfigUpdate, db: Session = Depends(get_db)):
    fields = {k: v for k, v in payload.model_dump(exclude={"api_key"}).items() if v is not None}
    # api_key 缺省/含掩码时保留原值，避免用掩码覆盖真实密钥
    api_key = payload.api_key or ""
    if api_key and "*" not in api_key:
        fields["api_key"] = api_key
    config_svc.update(db, **fields)
    return _config_out(db)


@router.get("/providers", response_model=list[ProviderPreset])
def list_providers():
    return [
        ProviderPreset(
            name=name,
            label=preset["label"],
            base_url=preset["base_url"],
            models=preset["models"],
            default_model=preset["default_model"],
        )
        for name, preset in PROVIDER_PRESETS.items()
    ]


@router.get("/status")
def status():
    return multimodal_loop.status()


@router.post("/analyze/{device_id}", response_model=AnalyzeResponse)
async def analyze_now(device_id: int):
    try:
        res = await multimodal_loop.analyze_once(device_id)
    except MultimodalError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AnalyzeResponse(
        code=0,
        device_id=device_id,
        snapshot=SnapshotOut(**res["snapshot"]),
        result=res["result"],
    )


@router.get("/results", response_model=list[SnapshotOut])
def list_results(
    device_id: int | None = None,
    event_type: str | None = None,
    severity: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(MonitoringSnapshot).order_by(MonitoringSnapshot.created_at.desc())
    if device_id is not None:
        stmt = stmt.where(MonitoringSnapshot.device_id == device_id)
    if event_type:
        stmt = stmt.where(MonitoringSnapshot.event_type == event_type)
    if severity:
        stmt = stmt.where(MonitoringSnapshot.severity == severity)
    stmt = stmt.limit(limit).offset(offset)
    return db.scalars(stmt).all()


@router.get("/results/{snapshot_id}/image")
def snapshot_image(snapshot_id: int, db: Session = Depends(get_db)):
    snp = db.get(MonitoringSnapshot, snapshot_id)
    if snp is None or not snp.snapshot_path:
        raise HTTPException(status_code=404, detail="快照不存在")
    from pathlib import Path

    path = Path(snp.snapshot_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="快照文件已清理")
    return FileResponse(str(path), media_type="image/jpeg")
