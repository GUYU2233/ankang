"""目标3回归测试：风险趋势聚合、老人风险档案、老人设备列表。

端点测试通过 dependency_overrides 将 get_db 指向隔离的临时 SQLite 库，
TestClient 不进入 lifespan，避免后台巡检循环写库干扰聚合断言。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, get_db
from app.main import app
from app.models.entities import AlertLevel, Device, Resident, RiskScore


@pytest.fixture()
def client_ctx(tmp_path):
    """隔离临时库 + 无 lifespan 的 TestClient + 测试 SessionLocal。"""
    eng = create_engine(
        f"sqlite:///{tmp_path / 'iso.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=eng)
    TestingSession = sessionmaker(bind=eng, autoflush=False, autocommit=False, future=True)

    def _override():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override
    try:
        yield TestClient(app), TestingSession
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_risk_trend(client_ctx) -> None:
    """GET /dashboard/risk-trend?days=7：7 个连续升序日期，今日聚合与手算一致。"""
    client, TestingSession = client_ctx
    now = datetime.now()
    today = now.replace(hour=12, minute=0, second=0, microsecond=0)
    three_days_ago = (now - timedelta(days=3)).replace(hour=12, minute=0, second=0, microsecond=0)
    with TestingSession() as db:
        resident = Resident(name="风险老人", gender="女")
        db.add(resident)
        db.flush()
        db.add_all([
            RiskScore(resident_id=resident.id, score=0.6, level=AlertLevel.YELLOW, factors_json="[]", created_at=today),
            RiskScore(resident_id=resident.id, score=0.8, level=AlertLevel.ORANGE, factors_json="[]", created_at=today),
            RiskScore(resident_id=resident.id, score=0.4, level=AlertLevel.YELLOW, factors_json="[]", created_at=three_days_ago),
        ])
        db.commit()

    r = client.get("/api/v1/dashboard/risk-trend?days=7")
    assert r.status_code == 200, r.text
    trend = r.json()
    assert len(trend) == 7
    dates = [p["date"] for p in trend]
    assert dates == sorted(dates)
    today_point = next(p for p in trend if p["date"] == now.strftime("%Y-%m-%d"))
    assert today_point["avg_score"] == 0.7
    assert today_point["max_score"] == 0.8
    assert today_point["count"] == 2
    three_point = next(p for p in trend if p["date"] == three_days_ago.strftime("%Y-%m-%d"))
    assert three_point["count"] == 1
    assert any(p["count"] == 0 for p in trend)  # 无数据日补 0

    # days=0 -> 422（ge=1 校验）
    r = client.get("/api/v1/dashboard/risk-trend?days=0")
    assert r.status_code == 422, r.text


def test_resident_risk_profile(client_ctx) -> None:
    """GET /residents/{id}/risk-profile：latest 取最新一条、factors 解析、trend 长度 7、404。"""
    client, TestingSession = client_ctx
    now = datetime.now()
    today = now.replace(hour=12, minute=0, second=0, microsecond=0)
    factors_json = (
        '[{"key":"body_lean","label":"躯干倾斜","value":0.2,"unit":"","normal_range":"0-0.4"},'
        '{"key":"support_base","label":"支撑面不稳","value":0.1,"unit":"","normal_range":"0-0.3"}]'
    )
    with TestingSession() as db:
        resident = Resident(name="档案老人", gender="男")
        db.add(resident)
        db.flush()
        db.add_all([
            RiskScore(resident_id=resident.id, score=0.5, level=AlertLevel.YELLOW, factors_json="[]", created_at=today - timedelta(days=1)),
            RiskScore(resident_id=resident.id, score=0.85, level=AlertLevel.RED, factors_json=factors_json, created_at=today),
        ])
        db.commit()
        resident_id = resident.id

    r = client.get(f"/api/v1/residents/{resident_id}/risk-profile?days=7")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["resident_id"] == resident_id
    assert data["resident_name"] == "档案老人"
    assert data["latest_score"] == 0.85
    assert data["latest_level"] == "red"
    assert data["updated_at"] is not None
    assert [f["label"] for f in data["factors"]] == ["躯干倾斜", "支撑面不稳"]
    assert data["factors"][0]["key"] == "body_lean"
    assert data["factors"][0]["value"] == 0.2
    assert len(data["trend"]) == 7
    today_point = next(p for p in data["trend"] if p["date"] == now.strftime("%Y-%m-%d"))
    assert today_point["count"] == 1
    assert today_point["avg_score"] == 0.85

    # 老人不存在 -> 404
    r = client.get("/api/v1/residents/99999/risk-profile")
    assert r.status_code == 404, r.text


def test_resident_devices(client_ctx) -> None:
    """GET /residents/{id}/devices：只返回该老人设备且含 resident_name；不存在 404。"""
    client, TestingSession = client_ctx
    with TestingSession() as db:
        resident = Resident(name="设备老人", gender="女")
        db.add(resident)
        db.flush()
        db.add_all([
            Device(device_name="客厅摄像头", device_serial="sim-dv-1", vendor="sim", scene="客厅", resident_id=resident.id),
            Device(device_name="卧室摄像头", device_serial="sim-dv-2", vendor="sim", scene="卧室", resident_id=resident.id),
            Device(device_name="门口摄像头", device_serial="sim-dv-3", vendor="sim", scene="门口"),  # 未绑定
        ])
        db.commit()
        resident_id = resident.id

    r = client.get(f"/api/v1/residents/{resident_id}/devices")
    assert r.status_code == 200, r.text
    items = r.json()
    assert len(items) == 2
    assert all(item["resident_id"] == resident_id for item in items)
    assert all(item["resident_name"] == "设备老人" for item in items)
    assert {i["device_serial"] for i in items} == {"sim-dv-1", "sim-dv-2"}

    # 老人不存在 -> 404
    r = client.get("/api/v1/residents/99999/devices")
    assert r.status_code == 404, r.text
