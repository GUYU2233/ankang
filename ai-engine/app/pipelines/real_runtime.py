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

    骨架估计：ultralytics YOLO26-Pose（yolo26s-pose.pt，可用 POSE_MODEL_PATH 覆盖，首次运行自动下载或放入
    ai-engine/models/ 目录）；跌倒判定：肩-髋中心高度与躯干倾角启发式；
    后续接入数据集训练 ST-GCN / TCN 后替换 fall_detect 方法。
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.getenv("POSE_MODEL_PATH", str(_BASE_DIR / "models" / "yolo26s-pose.pt"))
        self.temporal_model_path = os.getenv("TEMPORAL_MODEL_PATH", str(_BASE_DIR / "models" / "tcn_fall.onnx"))
        self.stgcn_model_path = os.getenv("STGCN_MODEL_PATH", str(_BASE_DIR / "runs" / "stgcn_binary_w32.onnx"))
        self._model = None
        self._temporal = None
        self._stgcn = None
        self.tcn_weight = float(os.getenv("TCN_WEIGHT", "0.65"))
        self.stgcn_weight = float(os.getenv("STGCN_WEIGHT", "0.35"))
        self._buffers: dict[str, deque[np.ndarray]] = {}
        self._anchors: dict[str, list] = {}  # 主目标 bbox 跨帧锚点 [x1,y1,x2,y2]
        self._track_boxes: dict[str, dict[int, np.ndarray]] = {}
        self._next_track_id: dict[str, int] = {}
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
        for key in [k for k in self._buffers if k == stream_id or k.startswith(stream_id + ':t')]:
            self._buffers.pop(key, None)
        self._anchors.pop(stream_id, None)
        self._track_boxes.pop(stream_id, None)
        self._next_track_id.pop(stream_id, None)
        self._risk_head.reset(stream_id)

    def reset_all_streams(self) -> None:
        self._buffers.clear()
        self._anchors.clear()
        self._track_boxes.clear()
        self._next_track_id.clear()
        self._risk_head.reset()

    def model_info(self) -> dict:
        pose = self._load_model()
        temporal = self._load_temporal_model()
        stgcn = self._load_stgcn_model()
        return {
            "pose_model": self.model_path,
            "pose_loaded": pose is not None,
            "temporal_model": self.temporal_model_path,
            "temporal_loaded": temporal is not None,
            "stgcn_model": self.stgcn_model_path,
            "stgcn_loaded": stgcn is not None,
            "fusion_weights": {"tcn": self.tcn_weight, "stgcn": self.stgcn_weight},
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

    def _load_stgcn_model(self):
        if self._stgcn is None:
            try:
                import onnxruntime as ort
                path = Path(self.stgcn_model_path)
                if not path.exists():
                    self._stgcn = False
                else:
                    self._stgcn = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                    logger.info(f"ST-GCN 时序模型已加载: {path}")
            except Exception as exc:
                logger.info(f"ST-GCN 时序模型不可用: {exc}")
                self._stgcn = False
        return self._stgcn if self._stgcn is not False else None

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
        sessions = []
        tcn = self._load_temporal_model()
        stgcn = self._load_stgcn_model()
        if tcn is not None:
            sessions.append((tcn, max(0.0, self.tcn_weight)))
        if stgcn is not None:
            sessions.append((stgcn, max(0.0, self.stgcn_weight)))
        if not sessions:
            return None
        buf = self._buffers.setdefault(stream_id, deque(maxlen=self.temporal_window))
        buf.append(self._normalize_skeleton(kpts))
        if len(buf) < self.temporal_window:
            return None
        seq = np.stack(list(buf), axis=0).astype(np.float32)
        x = np.transpose(seq, (2, 0, 1))[None, ...]
        weighted_logits = None
        total_weight = 0.0
        for session, weight in sessions:
            try:
                logits = np.asarray(session.run(None, {session.get_inputs()[0].name: x})[0][0], dtype=np.float32)
                if logits.shape != (2,) or not np.isfinite(logits).all():
                    continue
                weighted_logits = logits * weight if weighted_logits is None else weighted_logits + logits * weight
                total_weight += weight
            except Exception as exc:
                logger.warning(f"时序模型推理失败，已降级: {exc}")
        if weighted_logits is None or total_weight <= 0:
            return None
        logits = weighted_logits / total_weight
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

        all_kpts = results[0].keypoints.data.cpu().numpy()   # [N,17,3]
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return self._empty_result()
        all_boxes = results[0].boxes.xyxy.cpu().numpy()      # [N,4]

        # 过滤低质量虚检：单人场景模型可能误出多个低置信框
        # 1) 关键点有效数过滤：至少 MIN_VALID_KP 个关键点置信 >0.3 才算有效人体
        MIN_VALID_KP = 4
        valid_mask = (all_kpts[:, :, 2] > 0.3).sum(axis=1) >= MIN_VALID_KP
        if not valid_mask.any():
            return self._empty_result()
        all_kpts = all_kpts[valid_mask]
        all_boxes = all_boxes[valid_mask]
        # 2) 高重叠去重：IoU>0.85 视为同一人，保留关键点更多的那一个
        keep = list(range(len(all_kpts)))
        for i in range(len(all_boxes)):
            if i not in keep:
                continue
            for j in range(i + 1, len(all_boxes)):
                if j not in keep:
                    continue
                if self._iou(all_boxes[i], all_boxes[j]) > 0.85:
                    # 保留关键点有效数更多的
                    ki = int((all_kpts[i, :, 2] > 0.3).sum())
                    kj = int((all_kpts[j, :, 2] > 0.3).sum())
                    if ki >= kj:
                        keep.remove(j)
                    else:
                        keep.remove(i)
                        break
        all_kpts = all_kpts[keep]
        all_boxes = all_boxes[keep]
        n_person = int(all_kpts.shape[0])
        track_ids = self._assign_track_ids(stream_id, all_boxes)

        # 逐人几何跌倒扫描（主目标选择 + 他人跌倒兜底）
        geo_fall = [bool(self._fall_by_keypoints(all_kpts[i], frame_bgr.shape)[0]) for i in range(n_person)]

        prev_anchor = self._anchors.get(stream_id)
        main_idx = self._select_subject(all_boxes, geo_fall, stream_id, n_person)
        # 若主目标因跌倒发生了身份切换，清空时序窗口，避免混入不同人的骨架
        if geo_fall[main_idx] and prev_anchor is not None and self._iou(np.asarray(prev_anchor, dtype=np.float32), all_boxes[main_idx]) < 0.15:
            self._buffers.pop(stream_id, None)

        kpts = all_kpts[main_idx]
        xyxy = all_boxes[main_idx]
        others_bbox = [all_boxes[i].round(1).tolist() for i in range(n_person) if i != main_idx]

        fall_detected, fall_prob = self._fall_by_keypoints(kpts, frame_bgr.shape)
        model_nearfall = 0.0
        temporal = self._temporal_probs(kpts, f"{stream_id}:t{track_ids[main_idx]}")
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
        # 他人几何跌倒兜底：主目标未跌倒时，任何已检出的人体几何判跌也算跌倒
        others_fall = any(geo_fall[i] for i in range(n_person) if i != main_idx)
        fall_detected = fall_detected or others_fall
        if others_fall and fall_prob < 0.8:
            fall_prob = 0.8
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
            "person_count": n_person,
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
            "others_bbox": others_bbox,
            "track_id": track_ids[main_idx],
            "tracks": [{"track_id": track_ids[i], "bbox": all_boxes[i].round(1).tolist(), "fall_geometry": geo_fall[i]} for i in range(n_person)],
            "frame_ms": 0,
            "mock": False,
        }

    def _assign_track_ids(self, stream_id: str, boxes: np.ndarray) -> list[int]:
        """按 bbox IoU 贪心匹配，维持短时稳定 track id；每个检测最多匹配一次。"""
        previous = self._track_boxes.get(stream_id, {})
        assigned: list[int] = []
        used: set[int] = set()
        next_id = self._next_track_id.get(stream_id, 1)
        for box in boxes:
            candidates = [(self._iou(old_box, box), tid) for tid, old_box in previous.items() if tid not in used]
            best_iou, best_id = max(candidates, default=(0.0, -1))
            if best_iou >= 0.20:
                tid = best_id
            else:
                tid = next_id
                next_id += 1
            assigned.append(tid)
            used.add(tid)
        self._track_boxes[stream_id] = {tid: boxes[i].copy() for i, tid in enumerate(assigned)}
        self._next_track_id[stream_id] = next_id
        # 清理已消失轨迹的时序缓冲，防止长期累积。
        active = set(assigned)
        for key in [k for k in self._buffers if k.startswith(stream_id + ":t") and int(k.rsplit("t", 1)[1]) not in active]:
            self._buffers.pop(key, None)
        return assigned

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        return inter / (area_a + area_b - inter + 1e-6)

    def _select_subject(self, all_boxes: np.ndarray, geo_fall: list, stream_id: str, n: int) -> int:
        """多人场景主目标选择：跌倒者优先 > 与上一帧锚点 IoU 连续 > 最大 bbox 面积。"""
        # 1) 跌倒者优先
        for i in range(n):
            if geo_fall[i]:
                self._anchors[stream_id] = all_boxes[i].tolist()
                return i
        # 2) 与上一帧锚点保持连续
        anchor = self._anchors.get(stream_id)
        if anchor is not None:
            best_i, best_iou = 0, -1.0
            for i in range(n):
                iou = self._iou(np.asarray(anchor, dtype=np.float32), all_boxes[i])
                if iou > best_iou:
                    best_i, best_iou = i, iou
            if best_iou >= 0.15:
                self._anchors[stream_id] = all_boxes[best_i].tolist()
                return best_i
        # 3) 最大 bbox 面积兜底
        areas = [(float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])) for b in all_boxes]
        main_i = int(np.argmax(areas))
        self._anchors[stream_id] = all_boxes[main_i].tolist()
        return main_i

    def _fall_by_keypoints(self, kpts: np.ndarray, frame_shape: tuple) -> tuple[bool, float]:
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
            "others_bbox": [],
            "track_id": None,
            "tracks": [],
            "frame_ms": 0,
            "mock": False,
        }