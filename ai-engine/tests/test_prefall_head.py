"""前置风险头单测：不依赖 GPU / ONNX。

用法：python ai-engine/tests/test_prefall_head.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-engine"))

from app.pipelines.prefall_head import HeuristicPoseRiskHead, map_temporal_probs


FRAME = (720, 1280, 3)


def _kpts(shoulder_xy, hip_xy, ankle_xy=(None, None), lean_dx=0.0, conf=0.9) -> np.ndarray:
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 2] = conf
    sx, sy = shoulder_xy
    hx, hy = hip_xy
    k[5] = (sx - 30, sy, conf)
    k[6] = (sx + 30 + lean_dx, sy, conf)
    k[11] = (hx - 25, hy, conf)
    k[12] = (hx + 25, hy, conf)
    k[13] = (hx - 20, (sy + hy) / 2 + 40, conf)
    k[14] = (hx + 20, (sy + hy) / 2 + 40, conf)
    ax = hx if ankle_xy[0] is None else ankle_xy[0]
    ay = hy + 250 if ankle_xy[1] is None else ankle_xy[1]
    k[15] = (ax - 20, ay, conf)
    k[16] = (ax + 20, ay, conf)
    return k


def test_stagger_is_nearfall_not_fall() -> None:
    head = HeuristicPoseRiskHead(window=16)
    # 站立 + 高倾角
    pose = _kpts((360, 200), (360, 400), lean_dx=320)
    out = None
    for i in range(8):
        jitter = _kpts((360 + i, 200), (360, 400), ankle_xy=(360 + 8 * i, 650), lean_dx=320)
        out = head.update("s1", jitter, FRAME)
    assert out is not None
    assert out["nearfall_prob"] >= 0.45
    assert out["cues"]["stagger"] >= 0.45
    print("  stagger nearfall OK", out["nearfall_prob"], out["gait_unsteadiness"])


def test_hip_drop_recover_nearfall() -> None:
    head = HeuristicPoseRiskHead(window=16)
    standing = _kpts((360, 200), (360, 380), ankle_xy=(360, 640))
    dropped = _kpts((360, 280), (360, 520), ankle_xy=(360, 650))
    for _ in range(4):
        head.update("s2", standing, FRAME)
    for _ in range(3):
        head.update("s2", dropped, FRAME)
    out = None
    for _ in range(4):
        out = head.update("s2", standing, FRAME)
    assert out is not None
    assert out["nearfall_prob"] >= 0.35
    print("  hip-drop recover OK", out["nearfall_prob"], out["cues"])


def test_fallen_suppresses_nearfall() -> None:
    head = HeuristicPoseRiskHead(window=16)
    fallen = _kpts((360, 620), (360, 660), ankle_xy=(360, 690), lean_dx=40)
    out = None
    for _ in range(8):
        out = head.update("s3", fallen, FRAME)
    assert out is not None
    assert out["nearfall_prob"] < 0.25
    print("  fallen not nearfall OK", out["nearfall_prob"])


def test_standing_still_low_risk() -> None:
    head = HeuristicPoseRiskHead(window=16)
    pose = _kpts((360, 200), (360, 400), ankle_xy=(360, 650))
    out = None
    for _ in range(10):
        out = head.update("s4", pose, FRAME)
    assert out is not None
    assert out["nearfall_prob"] < 0.3
    assert out["gait_unsteadiness"] < 0.35
    print("  standing still OK", out)


def test_map_temporal_probs_splits_prefall() -> None:
    two = map_temporal_probs(np.array([0.7, 0.3], dtype=np.float32))
    assert abs(two["fall"] - 0.3) < 1e-5
    assert two["nearfall"] == 0.0  # 2 类 ONNX 不把 0.7 当 nearfall
    four = map_temporal_probs(np.array([0.1, 0.6, 0.2, 0.1], dtype=np.float32))
    assert abs(four["nearfall"] - 0.6) < 1e-5
    assert abs(four["risk_behavior"] - 0.2) < 1e-5
    print("  temporal softmax split OK")


def main() -> None:
    test_stagger_is_nearfall_not_fall()
    test_hip_drop_recover_nearfall()
    test_fallen_suppresses_nearfall()
    test_standing_still_low_risk()
    test_map_temporal_probs_splits_prefall()
    print("PREFALL HEAD OK")


if __name__ == "__main__":
    main()
