from __future__ import annotations

from datetime import datetime

import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core.alert_engine import AlertEngine
from app.db import Base
from app.models.entities import AlertEvent, AlertLevel, Device, FallEvent, NotificationLog


class _RiskResult:
    score = 0.95
    level = AlertLevel.RED
    factors = []
    events = ["fall"]
    trigger_reason = "test"


class _RiskEngine:
    def score(self, *args, **kwargs):
        return _RiskResult()

    @staticmethod
    def factors_json(_):
        return "[]"


class _Infer:
    def __init__(self, fall: bool):
        self.fall_detected = fall
        self.fall_prob = 0.96 if fall else 0.0
        self.nearfall_prob = 0.0
        self.gait_unsteadiness = 0.0
        self.fall_type = "pose_fall" if fall else ""
        self.mock = False
        self.risk_score = self.fall_prob


def _session(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'lifecycle.db'}")
    Base.metadata.create_all(eng)
    return sessionmaker(bind=eng, future=True)()


def test_fall_event_reuses_and_recovers(tmp_path):
    db = _session(tmp_path)
    device = Device(device_name="cam", device_serial="life-1", scene="卧室", enabled=True)
    db.add(device); db.commit(); db.refresh(device)
    engine = AlertEngine(_RiskEngine(), confirm_frames=1)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    assert engine.process(db, device, _Infer(True), 0.96, 100.0, frame) is not None
    assert len(db.scalars(select(FallEvent)).all()) == 1
    engine.process(db, device, _Infer(True), 0.96, 120.0, frame)
    assert len(db.scalars(select(FallEvent)).all()) == 1
    for i in range(3):
        engine.process(db, device, _Infer(False), 0.0, 130.0 + i)
    fall = db.scalar(select(FallEvent))
    assert fall.end_at is not None
    assert fall.screenshot_path


def test_alert_window_aggregates(tmp_path):
    db = _session(tmp_path)
    device = Device(device_name="cam", device_serial="agg-1", scene="卧室", enabled=True)
    db.add(device); db.commit(); db.refresh(device)
    engine = AlertEngine(_RiskEngine(), confirm_frames=1)
    engine.process(db, device, _Infer(True), 0.96, 100.0)
    engine._last_emit_ts[device.id] = 0
    engine.process(db, device, _Infer(True), 0.96, 120.0)
    assert len(db.scalars(select(AlertEvent)).all()) == 1
    assert len(db.scalars(select(NotificationLog)).all()) == 1
