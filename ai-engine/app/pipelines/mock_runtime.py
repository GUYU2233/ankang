from __future__ import annotations

import random
import time

import cv2
import numpy as np


class MockRuntime:
    """无模型时使用的模拟推理：基于画面简单统计 + 时间随机因子，保证系统闭环演示。"""

    def __init__(self) -> None:
        self.rng = random.Random(int(time.time() * 1000) % 100000)

    def infer(self, frame_bgr: np.ndarray) -> dict:
        # 画面统计：亮度、运动纹理强度近似
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean()) / 255.0
        edges = cv2.Canny(gray, 80, 160)
        texture = float((edges > 0).mean())

        # 检测偏蓝绿色块（模拟画面中的人体色块），估计目标大小
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (90, 60, 60), (140, 255, 255))
        ratio = float((mask > 0).mean())
        person_count = 1 if ratio > 0.005 else 0

        fall_roll = self.rng.random()
        fall_detected = person_count and fall_roll < 0.06
        if fall_detected:
            fall_prob = 0.75 + self.rng.random() * 0.2
            risk_score = 0.85 + self.rng.random() * 0.1
            fall_type = self.rng.choice(["forward_fall", "slip_fall", "fall_from_bed"])
        else:
            fall_prob = round(self.rng.random() * 0.2, 3)
            base = 0.25 + 0.25 * (1 - brightness) + 0.3 * texture
            risk_score = round(min(0.9, base + self.rng.random() * 0.25), 3)

        level = "green"
        if fall_detected or risk_score >= 0.85:
            level = "red"
        elif risk_score >= 0.7:
            level = "orange"
        elif risk_score >= 0.45:
            level = "yellow"

        factors = [
            {"key": "body_lean", "label": "躯干倾斜", "value": round(min(1.0, texture * 6 + self.rng.random() * 0.3), 2), "unit": "", "normal_range": "0-0.4"},
            {"key": "support_base", "label": "支撑面不稳", "value": round(min(1.0, texture * 4), 2), "unit": "", "normal_range": "0-0.3"},
            {"key": "posture_height", "label": "姿态高度异常", "value": round(min(1.0, self.rng.random() * 0.4), 2), "unit": "", "normal_range": "0-0.3"},
        ]

        return {
            "person_count": person_count,
            "fall_detected": fall_detected,
            "fall_prob": round(fall_prob, 3),
            "fall_type": fall_type,
            "risk_factors": factors,
            "risk_score": risk_score,
            "level": level,
            "frame_ms": 0,
            "mock": True,
        }