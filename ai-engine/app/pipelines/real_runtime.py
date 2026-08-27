from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from loguru import logger


class RealRuntime:
    """真实推理流水线。

    骨架估计：ultralytics YOLOv8-Pose（yolov8n-pose.pt，首次运行自动下载或放入
    ai-engine/models/ 目录）；跌倒判定：肩-髋中心高度与躯干倾角启发式；
    后续接入数据集训练 ST-GCN / TCN 后替换 fall_detect 方法。
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or "models/yolov8n-pose.pt"
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                import ultralytics
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
                logger.info(f"姿态模型已加载: {self.model_path}")
            except Exception as exc:
                logger.info(f"姿态模型不可用: {exc}")
                self._model = False
        return self._model if self._model is not False else None

    def infer(self, frame_bgr: np.ndarray) -> dict | None:
        model = self._load_model()
        if model is None:
            return None
        try:
            from ultralytics.engine.results import Results
            results: list[Results] = model.predict(frame_bgr, verbose=False, conf=0.35, device=0, half=False)
        except Exception as exc:
            logger.warning(f"姿态推理失败: {exc}")
            return None
        if not results or results[0].keypoints is None or results[0].keypoints.data.shape[0] == 0:
            return self._empty_result()

        kpts = results[0].keypoints.data[0].cpu().numpy()  # [17,3] xy,conf
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return self._empty_result()
        xyxy = boxes.xyxy.cpu().numpy()[0]
        fall_detected, fall_prob = self._fall_by_keypoints(kpts, frame_bgr.shape)
        risk_factors = self._risk_factors(kpts, frame_bgr.shape, fall_prob)
        risk_score = max(fall_prob, min(0.9, sum(float(f["value"]) for f in risk_factors) / len(risk_factors)))
        level = "green"
        if fall_detected or risk_score >= 0.85:
            level = "red"
        elif risk_score >= 0.7:
            level = "orange"
        elif risk_score >= 0.45:
            level = "yellow"
        return {
            "person_count": 1,
            "fall_detected": fall_detected,
            "fall_prob": round(float(fall_prob), 3),
            "fall_type": "pose_fall" if fall_detected else "",
            "risk_factors": risk_factors,
            "risk_score": round(risk_score, 3),
            "level": level,
            "frame_ms": 0,
            "mock": False,
        }

    def _fall_by_keypoints(self, kpts: np.ndarray, frame_shape: tuple) -> tuple[bool, float]:
        """根据骨架几何特征估计跌倒概率。"""
        h, w = frame_shape[:2]
        conf = kpts[:, 2]
        if conf[11] < 0.3 or conf[12] < 0.3 or conf[5] < 0.3 or conf[6] < 0.3:
            return False, 0.1
        shoulder_x = float((kpts[5, 0] + kpts[6, 0]) / 2)
        shoulder_y = float((kpts[5, 1] + kpts[6, 1]) / 2)
        hip_x = float((kpts[11, 0] + kpts[12, 0]) / 2)
        hip_y = float((kpts[11, 1] + kpts[12, 1]) / 2)
        ankle_y = float((kpts[15, 1] + kpts[16, 1]) / 2)
        hip_height_ratio = (h - hip_y) / max(h, 1e-6)
        body_height = abs(hip_y - shoulder_y) + 1e-6
        lean = abs(hip_x - shoulder_x) / body_height
        fall_prob = 0.0
        if hip_height_ratio < 0.18 or (hip_y - ankle_y) < body_height * 0.6:
            fall_prob = 0.8
        else:
            fall_prob = min(0.6, max(0.0, (lean - 0.8) * 2.0))
        return fall_prob > 0.55, fall_prob

    def _risk_factors(self, kpts: np.ndarray, frame_shape: tuple, fall_prob: float) -> list[dict]:
        h, w = frame_shape[:2]
        shoulder_x = float((kpts[5, 0] + kpts[6, 0]) / 2)
        shoulder_y = float((kpts[5, 1] + kpts[6, 1]) / 2)
        hip_x = float((kpts[11, 0] + kpts[12, 0]) / 2)
        hip_y = float((kpts[11, 1] + kpts[12, 1]) / 2)
        ankle_y = float((kpts[15, 1] + kpts[16, 1]) / 2)
        hip_ratio = max(0.0, min(1.0, (h - hip_y) / max(h, 1e-6)))
        body_h = max(abs(hip_y - shoulder_y), 1e-6)
        lean = abs(hip_x - shoulder_x) / body_h
        return [
            {"key": "gait_unsteady", "label": "步态不稳", "value": round(min(1.0, max(0.0, (lean - 0.55) * 2.5)), 2), "unit": "", "normal_range": "0-0.3"},
            {"key": "moving_speed", "label": "移动速度", "value": round(max(0.0, 0.6 - abs(hip_y - ankle_y) / max(h, 1e-6)), 2), "unit": "", "normal_range": "0-1"},
            {"key": "inactivity", "label": "长时间静止", "value": 0.1, "unit": "", "normal_range": "0-0.4"},
            {"key": "posture_stability", "label": "姿态稳定性", "value": round(max(0.0, min(1.0, 1.0 - lean)), 2), "unit": "", "normal_range": "0.7-1"},
        ]

    def _empty_result(self) -> dict:
        return {
            "person_count": 0,
            "fall_detected": False,
            "fall_prob": 0.0,
            "fall_type": "",
            "risk_factors": [],
            "risk_score": 0.0,
            "level": "green",
            "frame_ms": 0,
            "mock": False,
        }