"""前置风险评分：风险因子注册表 + 行为累积器 + 规则评分引擎。

约定：所有风险因子取值统一为 0-1，值越大风险越高。
- kinematic 因子由 ai-engine 逐帧输出；
- behavioral 因子由本模块按"老人+设备"维度跨帧累积（久坐、卫生间停留、夜间离床、步态波动）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.models.entities import AlertLevel


@dataclass(frozen=True)
class FactorSpec:
    key: str
    label: str
    weight: float
    normal_range: str = "0-0.3"
    unit: str = ""
    source: str = "kinematic"  # kinematic | behavioral
    description: str = ""


FACTOR_SPECS: dict[str, FactorSpec] = {
    "body_lean": FactorSpec("body_lean", "躯干倾斜", 0.16, "0-0.4", "", "kinematic", "肩髋中心横向偏移与躯干长度之比"),
    "support_base": FactorSpec("support_base", "支撑面不稳", 0.10, "0-0.3", "", "kinematic", "双踝间距相对躯干长度过窄"),
    "posture_height": FactorSpec("posture_height", "姿态高度异常", 0.20, "0-0.3", "", "kinematic", "髋部离地高度过低(疑似倒地/坐地)"),
    "inactivity": FactorSpec("inactivity", "长时间静止", 0.16, "0-0.4", "", "behavioral", "持续静止/久坐久卧"),
    "bathroom_dwell": FactorSpec("bathroom_dwell", "卫生间停留", 0.12, "0-0.4", "", "behavioral", "卫生间连续停留时间"),
    "night_trips": FactorSpec("night_trips", "夜间频繁离床", 0.10, "0-0.4", "", "behavioral", "夜间卧室出现次数"),
    "gait_unsteadiness": FactorSpec("gait_unsteadiness", "步态不稳(时序)", 0.16, "0-0.3", "", "behavioral", "躯干倾斜的时序波动"),
}


@dataclass
class RiskConfig:
    inactivity_minutes: float = 120.0
    bathroom_dwell_minutes: float = 30.0
    night_trip_count: int = 3
    night_start_hour: int = 22
    night_end_hour: int = 6
    still_risk_score_thr: float = 0.25
    gait_window: int = 30
    gait_std_thr: float = 0.25
    fall_floor: float = 0.90
    orange_score: float = 0.70
    yellow_score: float = 0.45
    red_score: float = 0.85


@dataclass
class RiskResult:
    score: float
    level: AlertLevel
    factors: list[dict[str, Any]]
    events: list[str]


def level_of(score: float, fall_detected: bool = False) -> AlertLevel:
    if fall_detected or score >= 0.85:
        return AlertLevel.RED
    if score >= 0.70:
        return AlertLevel.ORANGE
    if score >= 0.45:
        return AlertLevel.YELLOW
    return AlertLevel.GREEN


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def extract_kinematic(infer: Any) -> dict[str, float]:
    """从 ai-engine 推理结果抽取逐帧运动学风险因子。"""
    factors = getattr(infer, "risk_factors", None) or []
    out: dict[str, float] = {}
    for f in factors:
        if isinstance(f, dict):
            key = f.get("key")
            value = f.get("value", 0.0)
        else:
            key = getattr(f, "key", None)
            value = getattr(f, "value", 0.0)
        spec = FACTOR_SPECS.get(key or "")
        if spec and spec.source == "kinematic":
            out[key] = _clamp01(value)
    return out


def _fmt_minutes(minutes: float) -> str:
    if minutes >= 60:
        return f"{minutes / 60:.1f}小时"
    return f"{minutes:.0f}分钟"


class BehavioralTracker:
    """按 subject key 维护跨帧行为状态，输出 0-1 行为风险因子。"""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._states: dict[str, dict] = {}

    def update(self, key: str, scene: str, infer: Any, now: float, kin: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        cfg = self.config
        st = self._states.setdefault(key, {
            "present": False,
            "still_since": None,
            "bathroom_since": None,
            "night_trips": 0,
            "night_last_present": False,
            "lean_history": [],
        })

        present = int(getattr(infer, "person_count", 0) or 0) > 0
        fall = bool(getattr(infer, "fall_detected", False))
        risk_score = _clamp01(getattr(infer, "risk_score", 0.0) or 0.0)
        hour = datetime.fromtimestamp(now).hour
        is_night = hour >= cfg.night_start_hour or hour < cfg.night_end_hour
        scene_key = (scene or "").strip()

        # 长时间静止：有人、无跌倒、整体风险低，持续越久越接近 1。
        if present and not fall and risk_score < cfg.still_risk_score_thr:
            if st["still_since"] is None:
                st["still_since"] = now
        else:
            st["still_since"] = None
        still_min = (now - st["still_since"]) / 60.0 if st["still_since"] else 0.0
        inactivity = _clamp01(still_min / cfg.inactivity_minutes) if cfg.inactivity_minutes else 0.0

        # 卫生间停留。
        if present and scene_key == "卫生间":
            if st["bathroom_since"] is None:
                st["bathroom_since"] = now
        else:
            st["bathroom_since"] = None
        bath_min = (now - st["bathroom_since"]) / 60.0 if st["bathroom_since"] else 0.0
        bathroom_dwell = _clamp01(bath_min / cfg.bathroom_dwell_minutes) if cfg.bathroom_dwell_minutes else 0.0

        # 夜间离床：夜间卧室里出现→消失→再出现，每次出现计 1 次。
        if scene_key == "卧室" and is_night:
            if present and not st["night_last_present"]:
                st["night_trips"] += 1
        st["night_last_present"] = present
        night_trips = _clamp01(st["night_trips"] / cfg.night_trip_count) if cfg.night_trip_count else 0.0

        # 步态不稳：躯干倾斜在一段窗口内的标准差。
        lean = float(kin.get("body_lean", 0.0))
        st["lean_history"].append(lean)
        if len(st["lean_history"]) > cfg.gait_window:
            st["lean_history"].pop(0)
        gait = 0.0
        if len(st["lean_history"]) >= 2:
            m = sum(st["lean_history"]) / len(st["lean_history"])
            var = sum((x - m) ** 2 for x in st["lean_history"]) / len(st["lean_history"])
            gait = _clamp01(math.sqrt(var) / cfg.gait_std_thr) if cfg.gait_std_thr else 0.0

        st["present"] = present
        beh = {
            "inactivity": round(inactivity, 3),
            "bathroom_dwell": round(bathroom_dwell, 3),
            "night_trips": round(night_trips, 3),
            "gait_unsteadiness": round(gait, 3),
        }

        events: list[str] = []
        if inactivity >= 1.0:
            events.append(f"长时间静止({_fmt_minutes(still_min)})")
        if bathroom_dwell >= 1.0:
            events.append(f"卫生间停留超时({_fmt_minutes(bath_min)})")
        if st["night_trips"] >= cfg.night_trip_count:
            events.append(f"夜间频繁离床({st['night_trips']}次)")
        return beh, events


class RiskRuleEngine:
    """加权评分 + 规则升档。"""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def score(self, kin: dict[str, float], beh: dict[str, float], fall_detected: bool, fall_prob: float) -> RiskResult:
        factors: list[dict[str, Any]] = []
        weighted = 0.0
        total_w = 0.0
        for key, spec in FACTOR_SPECS.items():
            value = kin.get(key) if spec.source == "kinematic" else beh.get(key)
            if value is None:
                continue
            value = _clamp01(value)
            weighted += spec.weight * value
            total_w += spec.weight
            factors.append({
                "key": spec.key,
                "label": spec.label,
                "value": round(value, 3),
                "unit": spec.unit,
                "normal_range": spec.normal_range,
                "source": spec.source,
            })

        base = weighted / total_w if total_w else 0.0
        score = max(base, _clamp01(fall_prob) * 0.95)
        events: list[str] = []

        if fall_detected:
            score = max(score, self.config.fall_floor)
            events.append("疑似跌倒")

        # 规则升档：卫生间隔间停留 -> 橙；久坐/久卧、夜间频繁离床、步态不稳 -> 黄。
        if beh.get("bathroom_dwell", 0.0) >= 1.0:
            score = max(score, self.config.orange_score)
        if beh.get("inactivity", 0.0) >= 1.0 or beh.get("night_trips", 0.0) >= 1.0:
            score = max(score, self.config.yellow_score)
        if beh.get("gait_unsteadiness", 0.0) >= 0.8:
            score = max(score, self.config.yellow_score)

        score = round(_clamp01(score), 3)
        level = level_of(score, fall_detected)
        return RiskResult(score=score, level=level, factors=factors, events=events)
