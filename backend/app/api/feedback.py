"""反馈样本池：AI 只生成候选，人工审核后才可进入离线训练。"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.entities import AlertEvent, AlertFeedback

router = APIRouter(prefix="/feedback", tags=["自学习反馈"])


@router.get("/samples")
def list_samples(source: str | None = None, target: str | None = None, limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)):
    stmt = select(AlertFeedback).order_by(AlertFeedback.created_at.desc()).limit(limit)
    if source:
        stmt = stmt.where(AlertFeedback.source == source)
    if target:
        stmt = stmt.where(AlertFeedback.target == target)
    rows = db.scalars(stmt).all()
    return [{"id": x.id, "alert_id": x.alert_id, "device_id": x.device_id, "source": x.source, "target": x.target, "decision": x.decision, "confidence": x.confidence, "weight": x.weight, "operator": x.operator, "note": x.note, "snapshot_path": x.snapshot_path, "created_at": x.created_at} for x in rows]


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    rows = db.scalars(select(AlertFeedback)).all()
    return {
        "total": len(rows),
        "human": sum(x.source == "human" for x in rows),
        "vision_ai": sum(x.source == "vision_ai" for x in rows),
        "positive": sum(bool(x.decision) for x in rows),
        "training_ready": sum(x.source == "human" and bool(x.snapshot_path) for x in rows),
        "weights": {"human": 1.0, "detector": 0.65, "vision_ai": 0.35},
        "policy": "人工反馈覆盖AI；AI反馈仅作审核排序，不自动训练",
    }


@router.post("/export-candidates")
def export_candidates(db: Session = Depends(get_db)):
    """导出人工核验且有证据帧的清单；不生成伪 pose 标签，不自动训练。"""
    rows = db.scalars(select(AlertFeedback).where(AlertFeedback.source == "human").order_by(AlertFeedback.created_at)).all()
    candidates = []
    for x in rows:
        if not x.snapshot_path or not Path(x.snapshot_path).is_file():
            continue
        candidates.append({"feedback_id": x.id, "alert_id": x.alert_id, "device_id": x.device_id, "target": x.target, "decision": x.decision, "image_path": x.snapshot_path, "weight": 1.0, "requires_pose_review": True})
    root = Path(__file__).resolve().parents[3] / "data" / "feedback"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "candidates.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"schema_version": 1, "policy": "human_review_required_before_yolo_training", "samples": candidates}, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return {"count": len(candidates), "path": str(path.resolve()), "training_allowed": False, "next_step": "人工补齐17点pose标注并审核后，才可交给finetune_pose.py"}
