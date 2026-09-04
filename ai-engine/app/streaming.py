from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from loguru import logger

_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # 头面部
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),    # 肩臂
    (5, 11), (6, 12), (11, 12),                  # 躯干
    (11, 13), (12, 14), (13, 15), (14, 16),      # 腿
]
_LEVEL_COLORS = {
    "red": (0, 0, 255),
    "orange": (0, 140, 255),
    "yellow": (0, 215, 255),
    "green": (0, 200, 0),
}

# 追踪短暂丢失时，复用最近有效骨架的最大时长（秒）
_PERSIST_WINDOW_S = 1.5


def _draw_annotations(frame, result):
    """在原始帧上绘制人体包围框与 17 点骨架，多人场景额外画出其它人体。"""
    out = frame.copy()
    keypoints = result.get("keypoints") or []
    bbox = result.get("bbox") or []
    others_bbox = result.get("others_bbox") or []
    person_count = int(result.get("person_count") or 0)
    color = _LEVEL_COLORS.get(result.get("level", ""), (0, 200, 0))
    # 主目标：绿框 + 人数标签
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if person_count > 1:
            cv2.putText(out, f"people: {person_count}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1, cv2.LINE_AA)
    pts = []
    for p in keypoints[:17]:
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
    # 其它人体：淡灰框（不画骨架，避免画面杂乱）
    for ob in others_bbox:
        if len(ob) == 4:
            ox1, oy1, ox2, oy2 = (int(round(float(v))) for v in ob)
            cv2.rectangle(out, (ox1, oy1), (ox2, oy2), (190, 190, 190), 1)
    return out


@dataclass
class StreamState:
    stream_id: str
    source: str
    target_fps: float
    loop_file: bool = True
    latest_result: dict[str, Any] | None = None
    latest_frame: Any = None
    latest_at: float = 0.0
    last_pose_keypoints: list = field(default_factory=list)
    last_pose_bbox: list = field(default_factory=list)
    last_pose_level: str = "green"
    last_pose_at: float = 0.0
    last_pose_others: list = field(default_factory=list)
    last_pose_count: int = 0
    latest_jpeg: bytes | None = None
    jpeg_seq: int = 0
    frame_condition: threading.Condition = field(default_factory=threading.Condition)
    replay_buffer: deque = field(default_factory=deque)  # (timestamp, jpeg)
    recordings: dict[str, dict] = field(default_factory=dict)
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
        self.recording_dir = Path(__file__).resolve().parents[1] / "runtime" / "recordings"
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.prebuffer_seconds = 5.0
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
                        state.last_pose_keypoints = []
                        state.last_pose_bbox = []
                        state.last_pose_at = 0.0
                        state.last_pose_others = []
                        state.last_pose_count = 0
                        continue
                    state.error = "视频源读取结束或中断"
                    break
                result = self.runtime.execute(frame, stream_id=state.stream_id)
                now = time.time()
                state.latest_result = result
                state.latest_frame = frame
                state.latest_at = now
                draw_result = result
                if result.get("keypoints"):
                    state.last_pose_keypoints = result["keypoints"]
                    state.last_pose_bbox = result.get("bbox") or []
                    state.last_pose_level = result.get("level", "green")
                    state.last_pose_others = result.get("others_bbox") or []
                    state.last_pose_count = int(result.get("person_count") or 0)
                    state.last_pose_at = now
                elif state.last_pose_keypoints and (now - state.last_pose_at) <= _PERSIST_WINDOW_S:
                    # 短暂丢检仅用于画面平滑，不改动真实推理结果。
                    draw_result = {**result, "keypoints": state.last_pose_keypoints, "bbox": state.last_pose_bbox, "level": state.last_pose_level, "others_bbox": state.last_pose_others, "person_count": state.last_pose_count}
                annotated = _draw_annotations(frame, draw_result)
                ok_jpg, jpg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok_jpg:
                    with state.frame_condition:
                        state.latest_jpeg = jpg_buf.tobytes()
                        state.jpeg_seq += 1
                        state.frame_condition.notify_all()
                    state.replay_buffer.append((now, state.latest_jpeg))
                    cutoff = now - self.prebuffer_seconds
                    while state.replay_buffer and state.replay_buffer[0][0] < cutoff:
                        state.replay_buffer.popleft()
                    completed = []
                    for clip_id, rec in state.recordings.items():
                        rec["frames"].append((now, state.latest_jpeg))
                        if now >= rec["end_at"]:
                            completed.append((clip_id, rec))
                    for clip_id, rec in completed:
                        state.recordings.pop(clip_id, None)
                        threading.Thread(target=self._write_recording, args=(clip_id, state.stream_id, rec), daemon=True).start()
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

    def trigger_recording(self, stream_id: str, alert_id: str, post_seconds: float = 5.0) -> dict:
        """冻结告警前缓存并继续采集告警后帧，随后异步编码 MP4。"""
        with self._lock:
            state = self._states.get(stream_id)
        if state is None or not state.latest_jpeg:
            raise KeyError("stream not found or has no frames")
        clip_id = f"alert_{alert_id}_{uuid.uuid4().hex[:8]}"
        now = time.time()
        rec = {"alert_id": str(alert_id), "started_at": now, "end_at": now + max(1.0, min(30.0, post_seconds)), "fps": max(1.0, state.actual_fps or state.target_fps), "frames": list(state.replay_buffer), "status": "recording", "error": ""}
        state.recordings[clip_id] = rec
        self._write_manifest(clip_id, stream_id, rec)
        return self.recording_status(clip_id)

    def _manifest_path(self, clip_id: str) -> Path:
        return self.recording_dir / f"{clip_id}.json"

    def _video_path(self, clip_id: str) -> Path:
        return self.recording_dir / f"{clip_id}.mp4"

    def _write_manifest(self, clip_id: str, stream_id: str, rec: dict) -> None:
        payload = {k: v for k, v in rec.items() if k != "frames"}
        payload.update({"clip_id": clip_id, "stream_id": stream_id, "frame_count": rec.get("frame_count", len(rec.get("frames", []))), "video_path": str(self._video_path(clip_id)) if self._video_path(clip_id).is_file() else None})
        path = self._manifest_path(clip_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _write_recording(self, clip_id: str, stream_id: str, rec: dict) -> None:
        try:
            import imageio_ffmpeg
            frames = rec.get("frames", [])
            if not frames:
                raise RuntimeError("recording contains no frames")
            first = cv2.imdecode(__import__('numpy').frombuffer(frames[0][1], dtype=__import__('numpy').uint8), cv2.IMREAD_COLOR)
            if first is None:
                raise RuntimeError("failed to decode buffered frame")
            height, width = first.shape[:2]
            # H.264/yuv420p 需要偶数尺寸。
            width -= width % 2; height -= height % 2
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            out = self._video_path(clip_id)
            tmp = out.with_name(out.stem + ".tmp.mp4")
            cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(round(rec["fps"], 3)), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            assert proc.stdin is not None
            for _, jpeg in frames:
                image = cv2.imdecode(__import__('numpy').frombuffer(jpeg, dtype=__import__('numpy').uint8), cv2.IMREAD_COLOR)
                if image is None: continue
                if image.shape[1] != width or image.shape[0] != height:
                    image = cv2.resize(image, (width, height))
                proc.stdin.write(image.tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            code = proc.wait()
            if code != 0:
                raise RuntimeError(stderr[-500:] or f"ffmpeg exit {code}")
            tmp.replace(out)
            rec["frame_count"] = len(frames)
            rec["duration_seconds"] = round(len(frames) / max(1.0, rec["fps"]), 2)
            rec["status"] = "ready"
        except Exception as exc:
            rec["status"] = "failed"; rec["error"] = str(exc)
            logger.exception(f"回放片段 {clip_id} 编码失败: {exc}")
        finally:
            rec["frames"] = []
            self._write_manifest(clip_id, stream_id, rec)

    def recording_status(self, clip_id: str) -> dict:
        path = self._manifest_path(clip_id)
        if not path.is_file():
            raise KeyError("recording not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def recording_path(self, clip_id: str) -> Path | None:
        path = self._video_path(clip_id).resolve()
        root = self.recording_dir.resolve()
        return path if root in path.parents and path.is_file() else None

    def latest(self, stream_id: str) -> dict | None:
        with self._lock:
            state = self._states.get(stream_id)
            return dict(state.latest_result) if state and state.latest_result else None

    def frame_jpg(self, stream_id: str) -> bytes | None:
        """返回 worker 已编码的最新标注帧，避免每个客户端重复 JPEG 编码。"""
        with self._lock:
            state = self._states.get(stream_id)
        return bytes(state.latest_jpeg) if state and state.latest_jpeg else None

    def mjpeg(self, stream_id: str):
        """单槽低延迟 MJPEG 生成器：慢客户端自动丢旧帧。"""
        with self._lock:
            state = self._states.get(stream_id)
        if state is None:
            return
        last_seq = -1
        while not state.stop_event.is_set():
            with state.frame_condition:
                if state.jpeg_seq == last_seq:
                    state.frame_condition.wait(timeout=2.0)
                if state.latest_jpeg is None:
                    continue
                jpg = bytes(state.latest_jpeg)
                last_seq = state.jpeg_seq
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")

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
