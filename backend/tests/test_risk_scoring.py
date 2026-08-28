"""前置风险评分单元测试：静止、跌倒、卫生间停留、夜间离床、近跌融合。"""
from __future__ import annotations

from app.core.risk_engine import RiskEngine
from app.core.risk_scoring import RiskConfig, RiskRuleEngine, build_trigger_reason
from app.models.entities import AlertLevel


class FakeInfer:
    def __init__(
        self,
        person_count=1,
        fall=False,
        fall_prob=0.0,
        risk_score=0.1,
        factors=None,
        nearfall_prob=0.0,
        gait_unsteadiness=0.0,
    ):
        self.person_count = person_count
        self.fall_detected = fall
        self.fall_prob = fall_prob
        self.nearfall_prob = nearfall_prob
        self.gait_unsteadiness = gait_unsteadiness
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


def test_nearfall_raises_yellow_or_orange_without_fall() -> None:
    eng = _engine()
    inf = FakeInfer(
        fall=False,
        fall_prob=0.12,
        risk_score=0.2,
        factors=KIN,
        nearfall_prob=0.78,
        gait_unsteadiness=0.82,
    )
    result = eng.score("r3:d3", "客厅", inf, 3_000_000.0)
    assert result.level in (AlertLevel.YELLOW, AlertLevel.ORANGE)
    assert result.level != AlertLevel.RED
    assert result.score >= 0.45
    assert not any("疑似跌倒" in e for e in result.events)
    keys = {f["key"]: f["value"] for f in result.factors}
    assert keys.get("nearfall_prob", 0) >= 0.7
    assert keys.get("gait_unsteadiness", 0) >= 0.8
    assert "跌倒前兆" in result.trigger_reason or "步态不稳" in result.trigger_reason


def test_trigger_reason_combines_gait_and_bathroom() -> None:
    eng = _engine()
    now = 4_000_000.0
    result = None
    inf = FakeInfer(risk_score=0.35, factors=KIN, nearfall_prob=0.2, gait_unsteadiness=0.82)
    for i in range(20):
        result = eng.score("r4:d4", "卫生间", inf, now + i * 2)
    assert result is not None
    assert result.score >= 0.70
    assert "步态不稳" in result.trigger_reason
    assert "卫生间停留" in result.trigger_reason


def test_nearfall_from_infer_not_folded_into_fall() -> None:
    """高 nearfall_prob 不得把 fall_detected 当跌倒抬到红。"""
    rules = RiskRuleEngine()
    kin = {"body_lean": 0.2, "support_base": 0.1, "posture_height": 0.1, "nearfall_prob": 0.8}
    beh = {"inactivity": 0.0, "bathroom_dwell": 0.0, "night_trips": 0.0, "gait_unsteadiness": 0.5}
    result = rules.score(kin, beh, fall_detected=False, fall_prob=0.1)
    assert result.level == AlertLevel.ORANGE
    assert result.score >= 0.70
    assert result.score < 0.90


def test_build_trigger_reason_format() -> None:
    reason = build_trigger_reason(
        [
            {"key": "gait_unsteadiness", "label": "步态不稳", "value": 0.82},
            {"key": "bathroom_dwell", "label": "卫生间停留", "value": 1.0},
        ],
        ["卫生间停留超时(30分钟)"],
        fall_detected=False,
    )
    assert "步态不稳(0.82)" in reason
    assert "卫生间停留超时" in reason
