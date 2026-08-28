from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from app.core.sim_provider import SimCamera
from app.models.entities import Device
from app.schemas.schemas import AIInferResponse


@dataclass
class FramePacket:
    device_id: int
    frame: np.ndarray
    meta: dict[str, Any]
    captured_at: float


class StreamService:
    """统一取流服务：模拟流 / 本地视频 / RTSP。"""

    def __init__(self) -> None:
        self._sim_cams: dict[int, SimCamera] = {}
        self._video_caps: dict[int, cv2.VideoCapture] = {}

    def reset_device(self, device_id: int) -> None:
        self._sim_cams.pop(device_id, None)
        cap = self._video_caps.pop(device_id, None)
        if cap is not None:
            cap.release()

    def get_frame(self, device: Device) -> FramePacket:
        """取一帧。返回 (frame, meta)。"""
        if device.vendor == "sim":
            cam = self._sim_cams.get(device.id)
            if cam is None:
                cam = SimCamera(device.id, device.device_name, device.scene)
                self._sim_cams[device.id] = cam
            frame, meta = cam.tick()
            return FramePacket(device.id, frame, meta, time.time())

        if device.access_url and device.access_url.lower().startswith("rtsp://"):
            return FramePacket(device.id, self._read_opencv(device), {}, time.time())

        if device.access_url and device.access_url.endswith((".mp4", ".avi", ".mkv")):
            return FramePacket(device.id, self._read_opencv(device, True), {}, time.time())

        # 无可用源时，退回合成帧，保证系统闭环演示
        cam = self._sim_cams.get(device.id)
        if cam is None:
            cam = SimCamera(device.id, device.device_name, device.scene)
            self._sim_cams[device.id] = cam
        frame, meta = cam.tick()
        return FramePacket(device.id, frame, meta, time.time())

    def _read_opencv(self, device: Device, is_file: bool = False) -> np.ndarray:
        cap = self._video_caps.get(device.id)
        should_open = cap is None
        if cap is not None and is_file:
            should_open = False
        if should_open:
            cap = cv2.VideoCapture(device.access_url)
            if not cap.isOpened():
                logger.warning(f"打开视频源失败: {device.access_url}")
                raise RuntimeError(f"无法打开视频源 {device.access_url}")
            self._video_caps[device.id] = cap
        ok, frame = cap.read()
        if not ok and is_file:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame[:] = (52, 56, 68)
        return frame

    def encode_jpeg(self, frame: np.ndarray, quality: int = 80) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return b""
        return buf.tobytes()

    def build_demo_result(self, device: Device, meta: dict) -> AIInferResponse:
        """AI 引擎不可用时的本地兜底推理结果，服务不中断。"""
        return AIInferResponse(
            person_count=meta.get("person_count", 1),
            fall_detected=bool(meta.get("fall_detected")),
            fall_prob=float(meta.get("risk_score") or 0.0),
            nearfall_prob=float(meta.get("nearfall_prob") or 0.0),
            gait_unsteadiness=float(meta.get("gait_unsteadiness") or 0.0),
            fall_type="sim_fall" if meta.get("fall_detected") else "",
            risk_factors=meta.get("risk_factors", []),
            risk_score=float(meta.get("risk_score") or 0.0),
            level="green",
            frame_ms=0,
            mock=True,
        )


stream_service = StreamService()