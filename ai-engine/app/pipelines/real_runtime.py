from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

_BASE_DIR = Path(__file__).resolve().parents[2]  # ai-engine 目录

_DLL_HOOKS: list = []


def _register_cuda_dll_dir() -> None:
    # onnxruntime-gpu 的 CUDA EP 需要 cuDNN 9 / CUDA 12 运行库；
    # 复用 PyTorch 自带的 torch/lib 下的 DLL，免去单独安装 CUDA toolkit。
    try:
        import torch as _torch
        lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(lib):
            _DLL_HOOKS.append(os.add_dll_directory(lib))
    except Exception:
        pass


_register_cuda_dll_dir()


class RealRuntime:
    """真实推理流水线。

    骨架估计：ultralytics YOLOv8-Pose（yolov8n-pose.pt，首次运行自动下载或放入
    ai-engine/models/ 目录）；跌倒判定：肩-髋中心高度与躯干倾角启发式；
    后续接入数据集训练 ST-GCN / TCN 后替换 fall_detect 方法。
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or str(_BASE_DIR / "models" / "yolov8n-pose.pt")
        self.temporal_model_path = os.getenv("TEMPORAL_MODEL_PATH", str(_BASE_DIR / "models" / "tcn_fall.onnx"))
        self._model = None
        self._temporal = None
        self._buffers: dict[str, deque[np.ndarray]] = {}
        self.temporal_window = int(os.getenv("TEMPORAL_WINDOW", "32"))
        self.fall_threshold = float(os.getenv("FALL_THRESHOLD", "0.95"))
        from app.pipelines.prefall_head import HeuristicPoseRiskHead
        self._risk_head = HeuristicPoseRiskHead(window=16)

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

    def reset_stream(self, stream_id: str) -> None:
        self._buffers.pop(stream_id, None)
        self._risk_head.reset(stream_id)

    def reset_all_streams(self) -> None:
        self._buffers.clear()
        self._risk_head.reset()

    def model_info(self) -> dict:
        pose = self._load_model()
        temporal = self._load_temporal_model()
        return {
            "pose_model": self.model_path,
            "pose_loaded": pose is not None,
            "temporal_model": self.temporal_model_path,
            "temporal_loaded": temporal is not None,
            "temporal_window": self.temporal_window,
            "fall_threshold": self.fall_threshold,
            "onnx_providers": temporal.get_providers() if temporal else [],
        }

    def _load_temporal_model(self):
        if self._temporal is None:
            try:
                import onnxruntime as ort
                path = Path(self.temporal_model_path)
                if not path.exists():
                    self._temporal = False
                else:
                    self._temporal = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                    logger.info(f"时序模型已加载: {path}")
            except Exception as exc:
                logger.info(f"时序模型不可用: {exc}")
                self._temporal = False
        return self._temporal if self._temporal is not False else None

    @staticmethod
    def _normalize_skeleton(kpts: np.ndarray) -> np.ndarray:
        out = np.zeros_like(kpts, dtype=np.float32)
        conf = kpts[:, 2]
        valid = conf >= 0.3
        xy = kpts[:, :2]
        if valid[5] and valid[6] and valid[11] and valid[12]:
            shoulder = (xy[5] + xy[6]) / 2.0
            hip = (xy[11] + xy[12]) / 2.0
            center = hip
            scale = float(np.linalg.norm(shoulder - hip))
        elif valid.any():
            pts = xy[valid]
            center = pts.mean(axis=0)
            scale = float(np.max(np.linalg.norm(pts - center, axis=1)))
        else:
            return out
        out[:, :2] = (xy - center) / max(scale, 1e-6)
        out[:, 2] = conf
        out[~valid] = 0.0
        return out

    @staticmethod
    def _map_temporal_probs(probs: np.ndarray) -> dict[str, float]:
        from app.pipelines.prefall_head import map_temporal_probs
        return map_temporal_probs(probs)

    def _temporal_probs(self, kpts: np.ndarray, stream_id: str) -> dict[str, float] | None:
        session = self._load_temporal_model()
        if session is None:
            return None
        buf = self._buffers.setdefault(stream_id, deque(maxlen=self.temporal_window))
        buf.append(self._normalize_skeleton(kpts))
        if len(buf) < self.temporal_window:
            return None
        seq = np.stack(list(buf), axis=0).astype(np.float32)
        x = np.transpose(seq, (2, 0, 1))[None, ...]
        logits = session.run(None, {session.get_inputs()[0].name: x})[0][0]
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return self._map_temporal_probs(probs)

    def _temporal_fall(self, kpts: np.ndarray, stream_id: str) -> float | None:
        mapped = self._temporal_probs(kpts, stream_id)
        if mapped is None:
            return None
        return mapped["fall"]

    def infer(self, frame_bgr: np.ndarray, stream_id: str = "default") -> dict | None:
        model = self._load_model()
        if model is None:
            return None
        try:
            from ultralytics.engine.results import Results
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
            results: list[Results] = model.predict(frame_bgr, verbose=False, conf=0.35, device=device, half=False)
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
        model_nearfall = 0.0
        temporal = self._temporal_probs(kpts, stream_id)
        if temporal is not None:
            fall_prob = temporal["fall"]
            fall_detected = fall_prob >= self.fall_threshold
            model_nearfall = temporal["nearfall"]
        elif self._load_temporal_model() is not None:
            # 时序模型已加载但窗口未满：暖机期不触发跌倒，仅保留几何风险因子
            fall_detected = False
        head = self._risk_head.update(stream_id, kpts, frame_bgr.shape)
        nearfall_prob = max(model_nearfall, float(head.get("nearfall_prob") or 0.0))
        gait_unsteadiness = float(head.get("gait_unsteadiness") or 0.0)
        # 前置风险不得把 fall_detected 置位。
        risk_factors = self._risk_factors(kpts, frame_bgr.shape, fall_prob, nearfall_prob, gait_unsteadiness)
        mean_f = sum(float(f["value"]) for f in risk_factors) / max(len(risk_factors), 1)
        risk_score = max(float(fall_prob), min(0.82, float(nearfall_prob)), min(0.9, mean_f))
        level = "green"
        if fall_detected:
            level = "red"
        elif risk_score >= 0.7:
            level = "orange"
        elif risk_score >= 0.45:
            level = "yellow"
        return {
            "person_count": 1,
            "fall_detected": fall_detected,
            "fall_prob": round(float(fall_prob), 3),
            "nearfall_prob": round(float(nearfall_prob), 3),
            "gait_unsteadiness": round(float(gait_unsteadiness), 3),
            "fall_type": "pose_fall" if fall_detected else "",
            "risk_factors": risk_factors,
            "risk_score": round(float(risk_score), 3),
            "level": level,
            "keypoints": [[round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 2)] for p in kpts],
            "bbox": [round(float(v), 1) for v in xyxy],
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
        if hip_height_ratio < 0.18 or (ankle_y - hip_y) < body_height * 0.6:
            fall_prob = 0.8
        else:
            fall_prob = min(0.6, max(0.0, (lean - 0.8) * 2.0))
        return fall_prob > 0.55, fall_prob

    def _risk_factors(
        self,
        kpts: np.ndarray,
        frame_shape: tuple,
        fall_prob: float,
        nearfall_prob: float = 0.0,
        gait_unsteadiness: float = 0.0,
    ) -> list[dict]:
        h, w = frame_shape[:2]
        conf = kpts[:, 2]

        def mid(i: int, j: int) -> np.ndarray:
            return np.array([(kpts[i, 0] + kpts[j, 0]) / 2.0, (kpts[i, 1] + kpts[j, 1]) / 2.0])

        shoulder = mid(5, 6)
        hip = mid(11, 12)
        torso = float(np.linalg.norm(shoulder - hip)) or 1e-6

        lean = abs(float(hip[0] - shoulder[0])) / torso
        body_lean = round(min(1.0, lean), 2)

        if conf[15] >= 0.3 and conf[16] >= 0.3:
            ankle_sep = abs(float(kpts[15, 0] - kpts[16, 0]))
            support_base = round(min(1.0, max(0.0, 1.0 - ankle_sep / (0.8 * torso))), 2)
        else:
            support_base = 0.5

        hip_ratio = max(0.0, min(1.0, (h - float(hip[1])) / max(h, 1e-6)))
        posture_height = round(min(1.0, max(0.0, (0.40 - hip_ratio) / 0.40)), 2)

        return [
            {"key": "body_lean", "label": "躯干倾斜", "value": body_lean, "unit": "", "normal_range": "0-0.4"},
            {"key": "support_base", "label": "支撑面不稳", "value": support_base, "unit": "", "normal_range": "0-0.3"},
            {"key": "posture_height", "label": "姿态高度异常", "value": posture_height, "unit": "", "normal_range": "0-0.3"},
            {"key": "nearfall_prob", "label": "跌倒前兆", "value": round(min(1.0, max(0.0, float(nearfall_prob))), 3), "unit": "", "normal_range": "0-0.3"},
            {"key": "gait_unsteadiness", "label": "步态不稳", "value": round(min(1.0, max(0.0, float(gait_unsteadiness))), 3), "unit": "", "normal_range": "0-0.3"},
        ]

    def _empty_result(self) -> dict:
        return {
            "person_count": 0,
            "fall_detected": False,
            "fall_prob": 0.0,
            "nearfall_prob": 0.0,
            "gait_unsteadiness": 0.0,
            "fall_type": "",
            "risk_factors": [],
            "risk_score": 0.0,
            "level": "green",
            "keypoints": [],
            "bbox": [],
            "frame_ms": 0,
            "mock": False,
        }