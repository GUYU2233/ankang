from __future__ import annotations

import math
import time

import cv2
import numpy as np

from app.pipelines.mock_runtime import MockRuntime
from app.pipelines.real_runtime import RealRuntime


class AIRuntime:
    """推理运行时：优先尝试真实姿态模型，不可用时退回模拟推理。"""

    def __init__(self) -> None:
        self.real = RealRuntime()
        self.mock = MockRuntime()

    def execute(self, frame_bgr: np.ndarray) -> dict:
        started = time.time()
        result = self.real.infer(frame_bgr)
        if result is None:
            result = self.mock.infer(frame_bgr)
        result["frame_ms"] = int((time.time() - started) * 1000)
        result["mock"] = result.get("mock", False)
        return result


runtime = AIRuntime()