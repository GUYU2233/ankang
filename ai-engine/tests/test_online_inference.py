"""在线时序推理不变量测试：缓冲隔离、回退、归一化、模型信息。

用法：python ai-engine/tests/test_online_inference.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-engine"))
os.environ["TEMPORAL_MODEL_PATH"] = str(ROOT / "ai-engine" / "models" / "tcn_fall.onnx")

from app.pipelines.real_runtime import RealRuntime


def fake_kpts(conf: float = 0.9) -> np.ndarray:
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 2] = conf
    k[5] = (320, 200, conf)
    k[6] = (400, 200, conf)
    k[11] = (330, 400, conf)
    k[12] = (390, 400, conf)
    return k


def test_normalize_center_at_hip() -> None:
    rr = RealRuntime()
    out = rr._normalize_skeleton(fake_kpts())
    assert out.shape == (17, 3)
    assert abs(out[11, 0] + out[12, 0]) < 1e-4, "髋中心应归一化到原点"
    assert abs(out[11, 1] + out[12, 1]) < 1e-4, "髋中心 y 也应居中"
    print("  normalize OK")


def test_buffer_isolation_and_reset() -> None:
    rr = RealRuntime()
    rr._buffers["a"] = 1  # 占位，仅验证 reset 不影响其它流
    rr._temporal_fall(fake_kpts(), "s1")
    rr._temporal_fall(fake_kpts(), "s2")
    assert "s1" in rr._buffers and "s2" in rr._buffers
    rr.reset_stream("s1")
    assert "s1" not in rr._buffers
    assert "s2" in rr._buffers
    print("  buffer isolation/reset OK")


def test_window_incomplete_returns_none() -> None:
    rr = RealRuntime()
    rr.reset_all_streams()
    assert rr._temporal_fall(fake_kpts(), "w1") is None, "窗口未满应回退未激活"
    print("  incomplete-window fallback OK")


def test_model_info() -> None:
    rr = RealRuntime()
    info = rr.model_info()
    assert info["temporal_loaded"] is True
    assert info["temporal_window"] == 32
    print("  model_info OK:", info["temporal_model"])
    rr.reset_all_streams()


def main() -> None:
    test_normalize_center_at_hip()
    test_buffer_isolation_and_reset()
    test_window_incomplete_returns_none()
    test_model_info()
    print("ONLINE INFERENCE OK")


if __name__ == "__main__":
    main()
