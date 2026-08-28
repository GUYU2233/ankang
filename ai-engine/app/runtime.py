from __future__ import annotations

import time
import numpy as np

from app.pipelines.mock_runtime import MockRuntime
from app.pipelines.real_runtime import RealRuntime


class AIRuntime:
    """推理运行时：姿态检测 + 可选时序 ONNX 模型 + 模拟兜底。"""

    def __init__(self) -> None:
        self.real = RealRuntime()
        self.mock = MockRuntime()

    def execute(self, frame_bgr: np.ndarray, stream_id: str = "default") -> dict:
        started = time.time()
        result = self.real.infer(frame_bgr, stream_id=stream_id)
        if result is None:
            result = self.mock.infer(frame_bgr)
        result["frame_ms"] = int((time.time() - started) * 1000)
        result["mock"] = result.get("mock", False)
        return result

    def reset_stream(self, stream_id: str) -> None:
        self.real.reset_stream(stream_id)

    def reset_all_streams(self) -> None:
        self.real.reset_all_streams()

    def model_info(self) -> dict:
        return self.real.model_info()


runtime = AIRuntime()
