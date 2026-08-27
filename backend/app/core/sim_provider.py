from __future__ import annotations

import random
import time

import cv2
import numpy as np


class SimCamera:
    """合成室内监控画面，用于无摄像头的开发演示。

    person 为一个简笔人体矩形，状态机依次模拟 walking/idle/unsteady/falling/fallen，
    提供给前端的预览流与 AI 引擎推理使用，保证端到端闭环可演示。
    """

    STATES = ("walking", "idle", "unsteady", "falling", "fallen")

    def __init__(self, device_id: int, device_name: str = "", scene: str = "客厅", seed: int | None = None):
        self.device_id = device_id
        self.device_name = device_name
        self.scene = scene
        self.rng = random.Random(seed or (device_id * 97 + 11))
        self.frame_no = 0
        self.state = "walking"
        self.state_frames = 0
        self.fallen_frames = 0
        self.last_transition_at = time.time()
        self._base_person_h = 0.42

    def _transition(self) -> None:
        self.state_frames += 1
        r = self.rng.random()
        if self.state == "walking":
            if self.state_frames > 20 and r < 0.12:
                self.state = "unsteady"
        elif self.state == "unsteady":
            if self.state_frames > 6 and r < 0.30:
                self.state = "falling"
            elif self.state_frames > 12:
                self.state = "walking"
        elif self.state == "falling":
            if self.state_frames > 8:
                self.state = "fallen"
                self.fallen_frames = 0
        elif self.state == "fallen":
            self.fallen_frames += 1
            if self.fallen_frames > 25:
                if r < 0.25:
                    self.state = "walking"
                self.fallen_frames = 0
        elif self.state == "idle":
            if self.state_frames > 40 and r < 0.55:
                self.state = "walking"

    def tick(self) -> tuple[np.ndarray, dict]:
        """生成一帧 BGR 画面与模拟元数据。"""
        if self.state != "fallen" or self.rng.random() < 0.25:
            self._transition()
        frame = self._render()
        self.frame_no += 1
        return frame, self._meta()

    def _render(self) -> np.ndarray:
        h, w = 720, 1280
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (52, 56, 68)
        cv2.rectangle(img, (0, int(h * 0.55)), (w, h), (88, 92, 104), -1)
        cv2.putText(img, f"SIM-CAM {self.device_id} | {self.scene} | {self.device_name}", (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(img, f"state={self.state} frame={self.frame_no}", (24, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cx = w // 2 + int(80 * np.sin(self.frame_no / 14.0))
        floor_y = int(h * 0.88)
        bh, bw = 340, 150
        angle = 0.0
        if self.state == "unsteady":
            angle = 6.0 * np.sin(self.frame_no / 5.0)
        elif self.state == "falling":
            angle = 60.0
            bh, bw = 150, 320
        elif self.state == "fallen":
            bh, bw = 100, 380
            cx = int(w * 0.52)
            floor_y = int(h * 0.92)

        cx += int(40 * np.sin(self.frame_no / 8.0))
        color = (70, 170, 255)
        # 人体外接框，模拟肩-髋姿态
        pts = np.array([
            [cx - bw // 2, floor_y - bh // 3],
            [cx, floor_y - bh],
            [cx + bw // 2, floor_y - bh // 3],
            [cx + bw // 4, floor_y],
            [cx - bw // 4, floor_y],
        ], dtype=np.int32)
        M = cv2.getRotationMatrix2D((cx, floor_y), angle, 1.0)
        pts = cv2.transform(pts.reshape(1, -1, 2), M).reshape(-1, 2).astype(np.int32)
        cv2.fillConvexPoly(img, pts, color)
        bbox = cv2.boundingRect(pts)
        cv2.rectangle(img, bbox, (0, 255, 255), 2)
        cv2.putText(img, "PERSON", (bbox[0], max(90, bbox[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.drawContours(img, [pts], 0, (0, 0, 0), 2)
        return img

    def _meta(self) -> dict:
        fall_now = self.state in ("falling", "fallen")
        risk_score = 0.0
        if self.state == "unsteady":
            risk_score = 0.45
        elif self.state == "falling":
            risk_score = 0.85
        elif self.state == "fallen":
            risk_score = 0.95
        elif self.state == "idle":
            risk_score = 0.25

        factors = [
            {"key": "gait_unsteady", "label": "步态不稳", "value": round(0.9 if self.state in ("unsteady", "falling") else self.rng.random() * 0.2, 2), "unit": "", "normal_range": "0-0.3"},
            {"key": "moving_speed", "label": "移动速度", "value": round(0.2 + self.rng.random() * 0.5, 2), "unit": "", "normal_range": "0-1"},
            {"key": "inactivity", "label": "长时间静止", "value": round(0.7 if self.state == "idle" else 0.05, 2), "unit": "", "normal_range": "0-0.4"},
            {"key": "posture_stability", "label": "姿态稳定性", "value": round(0.1 if self.state in ("unsteady", "falling") else 0.9, 2), "unit": "", "normal_range": "0.7-1"},
        ]
        return {
            "state": self.state,
            "fall_detected": fall_now,
            "risk_score": round(risk_score, 2),
            "risk_factors": factors,
            "person_count": 1 if self.state != "fallen" else 1,
        }