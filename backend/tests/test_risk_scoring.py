"""前置风险评分单元测试：静止、跌倒、卫生间停留、夜间离床。"""
from __future__ import annotations

from app.core.risk_engine import RiskEngine
from app.core.risk_scoring import RiskConfig
from app.models.entities import AlertLevel


class FakeInfer:
    def __init__(self, person_count=1, fall=False, fall_prob=0.0, risk_score=0.1, factors=None):
        self.person_count = person_count
        self.fall_detected = fall
        self.fall_prob = fall_prob
        self.risk_score = risk_score
        self.risk_factors = factors or []
        self.fall_type = ""
        self.mock = False


KIN = [
    {"key": "body_lean", "label": "躯干倾斜", "value": 0.2, "unit": "", "normal_range": "0-0.4"},
    {"key": "support_base", "label": "支撑面不稳", "value": 0.1, "unit": "", "normal_range": "0-0.3"},
    {"key": "posture_height", "label": "姿态高度异常", "value": 0.1, "unit": "", "normal_range": "0-0.3"},
]


def _engine() -> RiskEngine:
    cfg = RiskConfig(inactivity_minutes=1.0, bathroom_dwell_minutes=0.5, night_trip_count=2)
    return RiskEngine(cfg)


def test_inactivity_escalation() -> None:
    eng = _engine()
    now = 1_000_000.0
    result = None
    for i in range(45):  # 90 秒 > 60 秒阈值
        result = eng.score("r1:d1", "客厅", FakeInfer(risk_score=0.1, factors=KIN), now + i * 2)
    assert result is not None
    assert result.score >= 0.45
    assert result.level in (AlertLevel.YELLOW, AlertLevel.ORANGE, AlertLevel.RED)
    assert any("长时间静止" in e for e in result.events)


def test_fall_is_red() -> None:
    eng = _engine()
    result = eng.score("r1:d1", "客厅", FakeInfer(fall=True, fall_prob=0.9, risk_score=0.95, factors=KIN), 1_000_000.0)
    assert result.level == AlertLevel.RED
    assert result.score >= 0.85


def test_bathroom_dwell_escalation() -> None:
    eng = _engine()
    now = 2_000_000.0
    result = None
    for i in range(20):  # 40 秒 > 30 秒阈值
        result = eng.score("r2:d2", "卫生间", FakeInfer(risk_score=0.4, factors=KIN), now + i * 2)
    assert result is not None
    assert result.score >= 0.70
    assert result.level in (AlertLevel.ORANGE, AlertLevel.RED)

