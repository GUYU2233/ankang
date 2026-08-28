"""独立前置风险头：从短时姿态缓冲估计 nearfall_prob 与 gait_unsteadiness。

在线 2 类 TCN/ONNX 只负责 fall_prob（softmax 的 fall 类）。近跌 / 风险行为
不再并入 non_fall。本模块是启发式实现，无需 GPU。

TODO: 用 training.models.PrefallRiskHead 训练 4 类时序头
(fall / nearfall / risk_behavior / normal) 并导出 ONNX 后，将
HeuristicPoseRiskHead.update 替换为该模型推理。对外接口保持：
    update(stream_id, kpts[17,3], frame_shape) -> {nearfall_prob, gait_unsteadiness, cues}
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
CONF_THR = 0.3

# 髋部离地比例：(frame_h - hip_y) / frame_h，越大越接近站立。
STANDING_HIP = 0.35
SITTING_HIP = 0.20
FALLEN_HIP = 0.16


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _mid(kpts: np.ndarray, i: int, j: int) -> np.ndarray | None:
    if float(kpts[i, 2]) < CONF_THR or float(kpts[j, 2]) < CONF_THR:
        return None
    return (kpts[i, :2] + kpts[j, :2]) / 2.0


def extract_pose_cues(kpts: np.ndarray, frame_shape: tuple) -> dict[str, float] | None:
    """从单帧 COCO 17 点抽取前置风险所需的几何量。关键点不足时返回 None。"""
    h = float(frame_shape[0] if frame_shape else 1.0) or 1.0
    shoulder = _mid(kpts, L_SHOULDER, R_SHOULDER)
    hip = _mid(kpts, L_HIP, R_HIP)
    if shoulder is None or hip is None:
        return None
    torso = float(np.linalg.norm(shoulder - hip)) or 1e-6
    hip_h = _clamp01((h - float(hip[1])) / h)
    lean = abs(float(hip[0] - shoulder[0])) / torso

    ankle_lx = ankle_rx = 0.0
    ankle_vis = 0
    if float(kpts[L_ANKLE, 2]) >= CONF_THR:
        ankle_lx = float(kpts[L_ANKLE, 0])
        ankle_vis += 1
    if float(kpts[R_ANKLE, 2]) >= CONF_THR:
        ankle_rx = float(kpts[R_ANKLE, 0])
        ankle_vis += 1

    hip_to_ankle = 0.0
    ankle_count = 0
    for idx in (L_ANKLE, R_ANKLE):
        if float(kpts[idx, 2]) >= CONF_THR:
            hip_to_ankle += abs(float(kpts[idx, 1]) - float(hip[1]))
            ankle_count += 1
    if ankle_count:
        hip_to_ankle /= ankle_count
        knee_bend = _clamp01(1.0 - hip_to_ankle / (2.2 * torso))
    else:
        knee_bend = 0.0

    torso_xy = (shoulder + hip) / 2.0
    return {
        "hip_h": hip_h,
        "lean": float(lean),
        "torso": torso,
        "torso_x": float(torso_xy[0]),
        "torso_y": float(torso_xy[1]),
        "ankle_lx": ankle_lx,
        "ankle_rx": ankle_rx,
        "ankle_vis": float(ankle_vis),
        "knee_bend": float(knee_bend),
        "standing": 1.0 if hip_h >= STANDING_HIP else 0.0,
    }


def _score_buffer(buf: list[dict[str, float]]) -> dict[str, Any]:
    if len(buf) < 2:
        last = buf[-1] if buf else {}
        return {
            "nearfall_prob": 0.0,
            "gait_unsteadiness": 0.0,
            "cues": last,
        }

    hip = [c["hip_h"] for c in buf]
    lean = [c["lean"] for c in buf]
    current = buf[-1]
    hip_now = current["hip_h"]
    lean_now = current["lean"]
    torso_ref = max(float(np.median([c["torso"] for c in buf])), 1e-6)

    # --- 步态不稳：倾斜波动 + 踝轨迹 + 躯干抖动 ---
    lean_std = float(np.std(lean))
    lean_gait = _clamp01(lean_std / 0.18)

    path = 0.0
    steps = 0
    for prev, cur in zip(buf[:-1], buf[1:]):
        if prev["ankle_vis"] >= 1 and cur["ankle_vis"] >= 1:
            path += abs(cur["ankle_lx"] - prev["ankle_lx"]) + abs(cur["ankle_rx"] - prev["ankle_rx"])
            steps += 1
    ankle_gait = _clamp01(path / (torso_ref * max(steps, 1) * 0.55)) if steps else 0.0

    jitter = float(np.std([c["torso_x"] for c in buf]) + np.std([c["torso_y"] for c in buf]))
    torso_gait = _clamp01(jitter / (torso_ref * 0.22))

    if steps:
        gait = _clamp01(0.45 * lean_gait + 0.30 * ankle_gait + 0.25 * torso_gait)
    else:
        gait = _clamp01(0.65 * lean_gait + 0.35 * torso_gait)

    # --- 跌倒前兆（未完成跌倒）：回升的快速沉髋 / 下蹲 / 快速坐下 / 站立踉跄 ---
    window = hip[-12:] if len(hip) >= 4 else hip
    peak = max(window)
    trough = min(window)
    drop = peak - trough
    recovered = hip_now - trough
    hip_drop_recover = 0.0
    if drop >= 0.10 and recovered >= 0.40 * drop and trough >= FALLEN_HIP:
        hip_drop_recover = _clamp01((drop - 0.08) / 0.22) * _clamp01(recovered / max(drop, 1e-6))

    recent = hip[-6:] if len(hip) >= 3 else hip
    sit_drop = max(recent) - recent[-1]
    sit_fast = 0.0
    if SITTING_HIP <= hip_now < STANDING_HIP and sit_drop >= 0.10 and hip_now >= FALLEN_HIP:
        sit_fast = _clamp01((sit_drop - 0.06) / 0.20)

    squat = 0.0
    if SITTING_HIP <= hip_now < 0.42 and current["knee_bend"] >= 0.40 and hip_now >= FALLEN_HIP:
        squat = _clamp01(0.35 + 0.65 * current["knee_bend"] + 0.2 * (STANDING_HIP - hip_now) / 0.20)

    stagger = 0.0
    if hip_now >= STANDING_HIP:
        stagger = _clamp01((lean_now - 0.28) / 0.50)

    nearfall = max(hip_drop_recover, squat, sit_fast, stagger)
    # 已经倒地则交给 fall_prob，不把完成跌倒标成前兆。
    if hip_now < FALLEN_HIP:
        nearfall *= 0.15
    nearfall = _clamp01(min(nearfall, 0.85))

    return {
        "nearfall_prob": round(nearfall, 3),
        "gait_unsteadiness": round(gait, 3),
        "cues": {
            "hip_h": round(hip_now, 3),
            "lean": round(lean_now, 3),
            "knee_bend": round(current["knee_bend"], 3),
            "hip_drop_recover": round(hip_drop_recover, 3),
            "squat": round(squat, 3),
            "sit_fast": round(sit_fast, 3),
            "stagger": round(stagger, 3),
        },
    }


def map_temporal_probs(probs: np.ndarray) -> dict[str, float]:
    """拆分时序 softmax，避免把 nearfall/risk_behavior 并进 non_fall。

    - 2 类（现网 ONNX）：index1=fall，nearfall 由独立风险头补齐；
    - 3 类：fall / nearfall / normal；
    - 4 类：fall / nearfall / risk_behavior / normal。
    """
    n = int(len(probs))
    if n <= 1:
        p = float(probs[0]) if n == 1 else 0.0
        return {"fall": p, "nearfall": 0.0, "risk_behavior": 0.0, "normal": max(0.0, 1.0 - p)}
    if n == 2:
        return {"fall": float(probs[1]), "nearfall": 0.0, "risk_behavior": 0.0, "normal": float(probs[0])}
    if n == 3:
        return {"fall": float(probs[0]), "nearfall": float(probs[1]), "risk_behavior": 0.0, "normal": float(probs[2])}
    return {
        "fall": float(probs[0]),
        "nearfall": float(probs[1]),
        "risk_behavior": float(probs[2]),
        "normal": float(probs[3]),
    }


class HeuristicPoseRiskHead:
    """按 stream_id 维护短时姿态缓冲的启发式前置风险头。"""

    def __init__(self, window: int = 16) -> None:
        self.window = window
        self._buffers: dict[str, deque[dict[str, float]]] = {}

    def reset(self, stream_id: str | None = None) -> None:
        if stream_id is None:
            self._buffers.clear()
        else:
            self._buffers.pop(stream_id, None)

    def update(self, stream_id: str, kpts: np.ndarray, frame_shape: tuple) -> dict[str, Any]:
        cues = extract_pose_cues(kpts, frame_shape)
        if cues is None:
            return {"nearfall_prob": 0.0, "gait_unsteadiness": 0.0, "cues": {}}
        buf = self._buffers.setdefault(stream_id, deque(maxlen=self.window))
        buf.append(cues)
        return _score_buffer(list(buf))
