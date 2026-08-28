"""组合A回归测试：告警处置状态机、状态筛选与富化、旧库迁移、WS 载荷兼容。

不依赖 ai-engine：端点测试通过 dependency_overrides 将 get_db 指向隔离的临时 SQLite 库，
TestClient 不进入 lifespan，避免后台巡检循环写库干扰断言，保证结果确定性。
"""
from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.alert_engine import AlertEngine
from app.db import Base, _migrate_sqlite, get_db
from app.main import app
from app.models.entities import AlertEvent, AlertLevel, Device, Resident

_OLD_ALERT_TABLE = """
CREATE TABLE alert_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_no VARCHAR(64) NOT NULL,
    device_id INTEGER NOT NULL,
    resident_id INTEGER,
    level VARCHAR(6) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    title VARCHAR(255) NOT NULL,
    detail_json TEXT,
    confirmed BOOLEAN NOT NULL,
    handled BOOLEAN NOT NULL,
    created_at DATETIME NOT NULL
)
"""


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


def test_migrate_old_db(tmp_path) -> None:
    """旧 schema 临时库迁移：新列齐全、布尔语义回填、索引存在、幂等。"""
    db_path = tmp_path / "old.db"
    eng = create_engine(f"sqlite:///{db_path}")
    with eng.begin() as conn:
        conn.execute(text(_OLD_ALERT_TABLE))
        conn.execute(
            text(
                "INSERT INTO alert_events (alert_no, device_id, resident_id, level, event_type, title, confirmed, handled, created_at) VALUES "
                "('a1', 1, 1, 'green', 'fall_risk', 't1', 1, 0, '2025-01-01 00:00:00'), "
                "('a2', 1, 1, 'green', 'fall_risk', 't2', 1, 1, '2025-01-01 00:00:00'), "
                "('a3', 1, 1, 'green', 'fall_risk', 't3', 0, 0, '2025-01-01 00:00:00')"
            )
        )
    _migrate_sqlite(eng)
    with eng.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(alert_events)"))}
        for name in ("status", "confirmed_by", "confirmed_at", "confirm_note", "handled_by", "handled_at", "handle_note", "closed_at"):
            assert name in cols, f"迁移后缺少列 {name}"
        rows = dict(conn.execute(text("SELECT alert_no, status FROM alert_events ORDER BY id")).fetchall())
        assert rows == {"a1": "confirmed", "a2": "handled", "a3": "pending"}, rows
        idx = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='index' AND name='ix_alert_events_status'")
        ).scalar()
        assert idx == "ix_alert_events_status"
    # 幂等：重复迁移不报错、不回填破坏数据
    _migrate_sqlite(eng)
    with eng.connect() as conn:
        rows = dict(conn.execute(text("SELECT alert_no, status FROM alert_events ORDER BY id")).fetchall())
        assert rows == {"a1": "confirmed", "a2": "handled", "a3": "pending"}


def test_alert_state_machine(client_ctx) -> None:
    """confirm -> handle -> close 状态流转，非法跳转 409。"""
    client, TestingSession = client_ctx
    with TestingSession() as db:
        alert = AlertEvent(
            alert_no="stm-" + uuid4().hex[:10],
            device_id=1,
            resident_id=None,
            level=AlertLevel.GREEN,
            event_type="fall_risk",
            title="状态机测试",
            confirmed=False,
            handled=False,
            status="pending",
            created_at=datetime.now(),
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)
        alert_id = alert.id

    # 跳过 confirm 直接 handle -> 409
    r = client.post(f"/api/v1/alerts/{alert_id}/handle", json={"operator": "李四", "note": "现场已搀扶"})
    assert r.status_code == 409, r.text

    # confirm
    r = client.post(f"/api/v1/alerts/{alert_id}/confirm", json={"operator": "张三", "note": "已核实"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "confirmed"
    assert data["confirmed_by"] == "张三"
    assert data["confirm_note"] == "已核实"

    # handle
    r = client.post(f"/api/v1/alerts/{alert_id}/handle", json={"operator": "李四", "note": "现场已搀扶"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "handled"
    assert data["handled_at"] is not None
    assert data["handled_by"] == "李四"

    # close
    r = client.post(f"/api/v1/alerts/{alert_id}/close", json={"operator": "王五"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "closed"
    assert data["closed_at"] is not None

    # closed 后再 confirm -> 409
    r = client.post(f"/api/v1/alerts/{alert_id}/confirm", json={"operator": "赵六"})
    assert r.status_code == 409, r.text


def test_status_filter_and_enrich(client_ctx) -> None:
    """GET /alerts?status=confirmed 过滤正确且富化字段齐全；ack 旧接口兼容。"""
    client, TestingSession = client_ctx
    with TestingSession() as db:
        resident = Resident(name="王奶奶", gender="女", guardian_phone="13800000001")
        db.add(resident)
        db.flush()
        device = Device(
            device_name="客厅摄像头",
            device_serial="sim-ef-" + uuid4().hex[:8],
            vendor="sim",
            scene="客厅",
            resident_id=resident.id,
            enabled=False,
        )
        db.add(device)
        db.flush()
        alert = AlertEvent(
            alert_no="ef-" + uuid4().hex[:10],
            device_id=device.id,
            resident_id=resident.id,
            level=AlertLevel.ORANGE,
            event_type="fall_risk",
            title="富化测试",
            confirmed=True,
            handled=False,
            status="confirmed",
            created_at=datetime.now(),
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)
        alert_id = alert.id

    r = client.get("/api/v1/alerts?status=confirmed")
    assert r.status_code == 200, r.text
    items = r.json()
    assert all(item["status"] == "confirmed" for item in items)
    mine = next(item for item in items if item["id"] == alert_id)
    assert mine["resident_name"] == "王奶奶"
    assert mine["guardian_phone"] == "13800000001"
    for key in ("status", "resident_name", "guardian_phone"):
        assert key in mine

    # 旧接口 ack 兼容：handled=true -> status=handled
    r = client.post(f"/api/v1/alerts/{alert_id}/ack?handled=true")
    assert r.status_code == 200, r.text
    with TestingSession() as db:
        assert db.get(AlertEvent, alert_id).status == "handled"


class FakeInfer:
    def __init__(self, fall=False, fall_prob=0.0, fall_type="", mock=False, nearfall_prob=0.0):
        self.fall_detected = fall
        self.fall_prob = fall_prob
        self.fall_type = fall_type
        self.mock = mock
        self.nearfall_prob = nearfall_prob


def test_ws_payload_has_guardian_phone(client_ctx) -> None:
    """AlertEngine._emit 广播载荷：新增 guardian_phone/status，旧键一个不少。"""
    _, TestingSession = client_ctx
    db = TestingSession()
    try:
        resident = Resident(name="李奶奶", guardian_phone="13800000001")
        db.add(resident)
        db.flush()
        device = Device(
            device_name="客厅摄像头",
            device_serial="sim-ws-" + uuid4().hex[:8],
            vendor="sim",
            scene="客厅",
            resident_id=resident.id,
        )
        db.add(device)
        db.flush()
        infer = FakeInfer(fall=False, fall_prob=0.6, fall_type="", mock=True, nearfall_prob=0.1)
        payload = AlertEngine()._emit(db, device, resident.id, AlertLevel.ORANGE, 0.6, infer, [], [], "步态不稳")
    finally:
        db.rollback()
        db.close()

    assert payload["guardian_phone"] == "13800000001"
    assert payload["status"] == "pending"
    old_keys = {
        "id", "alert_no", "level", "event_type", "title", "device_id", "device_name",
        "scene", "resident_id", "resident_name", "score", "fall_prob", "trigger_reason", "created_at",
    }
    assert old_keys <= set(payload.keys()), f"缺失旧键: {old_keys - set(payload.keys())}"
