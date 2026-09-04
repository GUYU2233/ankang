"""人工核验高权重、视觉 AI 低权重的反馈与训练门禁测试。"""
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.alerts import _fuse_feedback
from app.db import Base, get_db
from app.main import app
from app.models.entities import AlertEvent, AlertFeedback, AlertLevel, Device


@pytest.fixture()
def ctx(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'feedback.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng, future=True)
    with Session() as db:
        dev = Device(device_name="cam", device_serial="verify-1", scene="卧室", enabled=True)
        db.add(dev); db.flush()
        alert = AlertEvent(alert_no="verify-alert", device_id=dev.id, level=AlertLevel.RED, event_type="fall_event", title="疑似跌倒", detail_json='{"fall_prob": 0.9}', created_at=datetime.now())
        db.add(alert); db.commit(); ids=(dev.id, alert.id)
    def override():
        with Session() as db: yield db
    app.dependency_overrides[get_db]=override
    yield TestClient(app), Session, ids
    app.dependency_overrides.pop(get_db,None)


def test_human_verification_has_final_authority(ctx):
    client, Session, (_, alert_id)=ctx
    res=client.post(f"/api/v1/alerts/{alert_id}/verify",json={"operator":"测试员","decision":False,"target":"fall"})
    assert res.status_code==200
    assert res.json()["status"]=="handled"
    with Session() as db:
        rows=db.query(AlertFeedback).all()
        fb=next(x for x in rows if x.source == "human")
        detector=next(x for x in rows if x.source == "detector")
        assert fb.weight==1.0 and fb.decision is False
        assert detector.weight==0.65
    human=AlertFeedback(decision=False,confidence=1.0,weight=1.0)
    ai=AlertFeedback(decision=True,confidence=0.99,weight=0.35)
    score,decision=_fuse_feedback(0.99,ai,human)
    assert score==0.0 and decision is False


def test_ai_only_never_training_ready(ctx, tmp_path):
    client, Session, (device_id, alert_id)=ctx
    with Session() as db:
        db.add(AlertFeedback(alert_id=alert_id,device_id=device_id,source="vision_ai",target="fall",decision=True,confidence=.9,weight=.35,snapshot_path=str(tmp_path/"x.jpg")))
        db.commit()
    result=client.post("/api/v1/feedback/export-candidates").json()
    assert result["count"]==0 and result["training_allowed"] is False
