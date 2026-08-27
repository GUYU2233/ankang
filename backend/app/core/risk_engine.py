from __future__ import annotations

import json
import time
from collections import defaultdict, deque

from loguru import logger

from app.models.entities import AlertLevel


def level_of(score: float, fall_detected: bool = False) -> AlertLevel:
    if fall_detected or score >= 0.85:
        return AlertLevel.RED
    if score >= 0.70:
        return AlertLevel.ORANGE
    if score >= 0.45:
        return AlertLevel.YELLOW
    return AlertLevel.GREEN


class RiskEngine:
    """跌倒风险前置预判引擎（滑动窗口评分）。"""

    def __init__(self, window_size: int = 12) -> None:
        self.window_size = window_size
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    def push_and_score(self, subject_key: str, score: float) -> float:
        dq = self._windows[subject_key]
        dq.append(float(score))
        while len(dq) > self.window_size:
            dq.popleft()
        if not dq:
            return 0.0
        recent = list(dq)[-5:] if len(dq) >= 5 else list(dq)
        return round(sum(recent) / len(recent), 3)

    def factors_json(self, infer) -> str:
        try:
            return json.dumps([f.model_dump() if hasattr(f, "model_dump") else f for f in infer.risk_factors], ensure_ascii=False)
        except Exception:
            return "[]"


risk_engine = RiskEngine()