"""统一取流服务：模拟流 / 本地视频（循环播放）/ RTSP。

支持模拟设备（vendor=sim）通过 access_url 指定本地视频文件并循环播放，
未指定时回落到合成画面。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from app.core.sim_provider import SimCamera
from app.models.entities import Device
from app.schemas.schemas import AIInferResponse

# 仓库根目录（backend/app/core/stream_service.py -> 向上三级）
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_VIDEO_DIR = _REPO_ROOT / "data" / "videos"
_VIDEO_EXTS = (".mp4", ".avi", ".mkv")
_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16),
]
_LEVEL_BGR = {"red": (0, 0, 255), "orange": (0, 140, 255), "yellow": (0, 215, 255), "green": (0, 200, 0)}


def _draw_skeleton(frame: np.ndarray, keypoints: list, bbox: list, level: str) -> np.ndarray:
    out = frame.copy()
    color = _LEVEL_BGR.get(level or "", (0, 200, 0))
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    pts = []
    for p in (keypoints or [])[:17]:
        if len(p) >= 3 and float(p[2]) >= 0.3:
            pts.append((int(round(float(p[0]))), int(round(float(p[1]))), float(p[2])))
        else:
            pts.append(None)
    for pt in pts:
        if pt is not None:
            cv2.circle(out, (pt[0], pt[1]), 3, color, -1)
    for a, b in _SKELETON_EDGES:
        if a < len(pts) and b < len(pts) and pts[a] is not None and pts[b] is not None:
            cv2.line(out, (pts[a][0], pts[a][1]), (pts[b][0], pts[b][1]), (0, 255, 255), 2)
    return out


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
        self._video_paths: dict[int, str] = {}
        self._latest_results: dict[int, AIInferResponse] = {}
        self._video_lock = threading.RLock()

    def reset_device(self, device_id: int) -> None:
        self._sim_cams.pop(device_id, None)
        cap = self._video_caps.pop(device_id, None)
        if cap is not None:
            cap.release()
        self._video_paths.pop(device_id, None)
        self._latest_results.pop(device_id, None)

    def set_latest_result(self, device_id: int, result: AIInferResponse | None) -> None:
        if result is not None:
            self._latest_results[device_id] = result
        else:
            self._latest_results.pop(device_id, None)

    def get_latest_result(self, device_id: int) -> AIInferResponse | None:
        return self._latest_results.get(device_id)

    def draw_overlay(self, frame: np.ndarray, infer: AIInferResponse) -> np.ndarray:
        if infer is None or not infer.keypoints:
            return frame
        return _draw_skeleton(frame, infer.keypoints, infer.bbox, infer.level)

    def resolve_video_path(self, url: str) -> str:
        """把本地视频地址解析为绝对路径，兼容多种写法；RTSP/HTTP 原样返回。"""
        if not url:
            return ""
        u = str(url).strip().strip('"').strip("'")
        if not u or u.lower().startswith(("rtsp://", "http://", "https://")):
            return u
        p = Path(u)
        if p.is_absolute():
            return str(p)
        candidates = [p, _REPO_ROOT / u, _DATA_VIDEO_DIR / u, _DATA_VIDEO_DIR / p.name]
        for c in candidates:
            try:
                if c.is_file():
                    return str(c.resolve())
            except OSError:
                continue
        # 兜底：按文件名在 data/videos 下递归查找
        if p.name:
            try:
                for hit in _DATA_VIDEO_DIR.rglob(p.name):
                    if hit.is_file():
                        return str(hit.resolve())
            except OSError:
                pass
        return str((_DATA_VIDEO_DIR / p.name).resolve())

    def list_local_videos(self) -> list[dict]:
        """列出 data/videos 下可用的本地视频（相对仓库根的路径）。"""
        items: list[dict] = []
        if not _DATA_VIDEO_DIR.is_dir():
            return items
        try:
            files = [f for f in _DATA_VIDEO_DIR.rglob("*") if f.is_file()]  # noqa: S612
        except OSError:
            return items
        for f in sorted(files):
            if f.suffix.lower() in _VIDEO_EXTS:
                rel = f.relative_to(_REPO_ROOT)
                try:
                    size = f.stat().st_size
                except OSError:
                    size = 0
                items.append({
                    "path": str(rel).replace("\\", "/"),
                    "name": f.name,
                    "scene": (f.parent.parent.name if f.parent.parent != _DATA_VIDEO_DIR else ""),
                    "size": size,
                })
        return items

    def get_frame(self, device: Device) -> FramePacket:
        """取一帧。本地视频/RTSP 优先于模拟合成画面（模拟设备也可指定视频）。"""
        url = (device.access_url or "").strip()

        if url.lower().startswith("rtsp://"):
            return FramePacket(device.id, self._read_opencv(device), {"source": "rtsp", "state": "streaming"}, time.time())

        if url.lower().endswith(_VIDEO_EXTS):
            return FramePacket(device.id, self._read_opencv(device, is_file=True), {"source": "local_video", "state": "playing"}, time.time())

        # 无视频源时：模拟设备生成合成画面；其它设备也退回合成帧保证闭环
        cam = self._sim_cams.get(device.id)
        if cam is None:
            cam = SimCamera(device.id, device.device_name, device.scene)
            self._sim_cams[device.id] = cam
        frame, meta = cam.tick()
        return FramePacket(device.id, frame, meta, time.time())

    def _read_opencv(self, device: Device, is_file: bool = False) -> np.ndarray:
        # OpenCV VideoCapture 非线程安全，多线程并发 read() 会触发 FFmpeg 断言崩溃，串行化访问。
        with self._video_lock:
            resolved = self.resolve_video_path(device.access_url or "")
            cached = self._video_paths.get(device.id)
            cap = self._video_caps.get(device.id)

            if cap is None or (cached is not None and cached != resolved):
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(resolved)
                if not cap.isOpened():
                    cap.release()
                    self._video_caps.pop(device.id, None)
                    self._video_paths.pop(device.id, None)
                    logger.warning(f"打开视频源失败: {resolved}")
                    raise RuntimeError(f"无法打开视频源 {resolved}")
                self._video_caps[device.id] = cap
                self._video_paths[device.id] = resolved

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
