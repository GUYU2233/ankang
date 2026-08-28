from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
from loguru import logger


@dataclass
class StreamState:
    stream_id: str
    source: str
    target_fps: float
    loop_file: bool = True
    latest_result: dict[str, Any] | None = None
    latest_at: float = 0.0
    frames_processed: int = 0
    started_at: float = field(default_factory=time.time)
    actual_fps: float = 0.0
    error: str = ""
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class StreamManager:
    """在 AI 引擎侧持续解码视频源并按目标 FPS 执行时序推理。"""

    def __init__(self, runtime, default_fps: float = 15.0) -> None:
        self.runtime = runtime
        self.default_fps = default_fps
        self._states: dict[str, StreamState] = {}
        self._lock = threading.RLock()

    def start(self, stream_id: str, source: str, target_fps: float | None = None, loop_file: bool = True) -> dict:
        fps = max(1.0, min(30.0, float(target_fps or self.default_fps)))
        with self._lock:
            current = self._states.get(stream_id)
            if current and current.source == source and current.thread and current.thread.is_alive():
                return self.status(stream_id)
            if current:
                self._stop_locked(current)
            state = StreamState(stream_id=stream_id, source=source, target_fps=fps, loop_file=loop_file)
            state.thread = threading.Thread(target=self._run, args=(state,), daemon=True, name=f"ai-stream-{stream_id}")
            self._states[stream_id] = state
            state.thread.start()
        return self.status(stream_id)

    def _run(self, state: StreamState) -> None:
        cap = cv2.VideoCapture(state.source)
        if not cap.isOpened():
            state.error = f"无法打开视频源: {state.source}"
            logger.warning(state.error)
            return
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        is_file = not state.source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        interval = 1.0 / state.target_fps
        next_tick = time.perf_counter()
        report_started = time.perf_counter()
        report_frames = 0
        try:
            while not state.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if is_file and state.loop_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.runtime.reset_stream(state.stream_id)
                        continue
                    state.error = "视频源读取结束或中断"
                    break
                result = self.runtime.execute(frame, stream_id=state.stream_id)
                now = time.time()
                state.latest_result = result
                state.latest_at = now
                state.frames_processed += 1
                report_frames += 1
                elapsed = time.perf_counter() - report_started
                if elapsed >= 1.0:
                    state.actual_fps = report_frames / elapsed
                    report_started = time.perf_counter()
                    report_frames = 0
                next_tick += interval
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    state.stop_event.wait(delay)
                else:
                    next_tick = time.perf_counter()
                if is_file and source_fps > state.target_fps:
                    # OpenCV 文件解码由目标节拍约束；不额外跳帧，确保时序连续。
                    pass
        except Exception as exc:
            state.error = str(exc)
            logger.exception(f"流 {state.stream_id} 推理异常: {exc}")
        finally:
            cap.release()

    def latest(self, stream_id: str) -> dict | None:
        with self._lock:
            state = self._states.get(stream_id)
            return dict(state.latest_result) if state and state.latest_result else None

    def status(self, stream_id: str) -> dict:
        with self._lock:
            state = self._states.get(stream_id)
            if state is None:
                return {"stream_id": stream_id, "running": False}
            return {
                "stream_id": stream_id,
                "source": state.source,
                "target_fps": state.target_fps,
                "actual_fps": round(state.actual_fps, 2),
                "frames_processed": state.frames_processed,
                "latest_at": state.latest_at,
                "running": bool(state.thread and state.thread.is_alive()),
                "error": state.error,
            }

    def health(self) -> dict:
        with self._lock:
            statuses = [self.status(k) for k in sorted(self._states)]
        return {"active_streams": sum(1 for s in statuses if s["running"]), "streams": statuses}

    def stop(self, stream_id: str) -> None:
        with self._lock:
            state = self._states.pop(stream_id, None)
            if state:
                self._stop_locked(state)
        self.runtime.reset_stream(stream_id)

    def _stop_locked(self, state: StreamState) -> None:
        state.stop_event.set()
        if state.thread and state.thread.is_alive():
            state.thread.join(timeout=3.0)

    def close(self) -> None:
        with self._lock:
            ids = list(self._states)
        for stream_id in ids:
            self.stop(stream_id)
