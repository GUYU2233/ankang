from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any

from app.core.risk_scoring import (
    BehavioralTracker,
    RiskConfig,
    RiskResult,
    RiskRuleEngine,
    _clamp01,
    build_trigger_reason,
    extract_kinematic,
    level_of,
)

__all__ = ["RiskEngine", "RiskConfig", "RiskResult", "level_of"]


class RiskEngine:
    """跌倒风险前置预判引擎：逐帧运动学因子 + 跨帧行为因子 + 规则评分。"""

    def __init__(self, config: RiskConfig | None = None, window_size: int = 12) -> None:
        self.config = config or RiskConfig()
        self.window_size = window_size
        self.tracker = BehavioralTracker(self.config)
        self.rules = RiskRuleEngine(self.config)
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    def score(self, subject_key: str, scene: str, infer: Any, now_ts: float) -> RiskResult:
        """综合评分：返回 RiskResult(score, level, factors, events)。"""
        kin = extract_kinematic(infer)
        beh, beh_events = self.tracker.update(subject_key, scene, infer, now_ts, kin)
        gait_ai = getattr(infer, "gait_unsteadiness", None)
        if gait_ai is not None:
            beh["gait_unsteadiness"] = max(beh.get("gait_unsteadiness", 0.0), _clamp01(gait_ai))
        fall = bool(getattr(infer, "fall_detected", False))
        fall_prob = float(getattr(infer, "fall_prob", 0.0) or 0.0)
        result = self.rules.score(kin, beh, fall, fall_prob)
        result.events = list(beh_events) + list(result.events)
        result.trigger_reason = build_trigger_reason(result.factors, result.events, fall)
        return result

    def push_and_score(self, subject_key: str, score: float) -> float:
        """保留的滑动窗口平滑接口（兼容旧调用）。"""
        dq = self._windows[subject_key]
        dq.append(float(score))
        while len(dq) > self.window_size:
            dq.popleft()
        if not dq:
            return 0.0
        recent = list(dq)[-5:] if len(dq) >= 5 else list(dq)
        return round(sum(recent) / len(recent), 3)

    def factors_json(self, factors: list[dict] | Any) -> str:
        try:
            if hasattr(factors, "risk_factors"):
                factors = factors.risk_factors
            data = []
            for f in factors or []:
                if isinstance(f, dict):
                    data.append(f)
                elif hasattr(f, "model_dump"):
                    data.append(f.model_dump())
                else:
                    data.append(vars(f))
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return "[]"


risk_engine = RiskEngine()
