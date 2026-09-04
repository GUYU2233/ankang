from __future__ import annotations

import json
import time
from collections import deque

import cv2
import numpy as np

from app.streaming import StreamManager, StreamState


class DummyRuntime:
    def reset_stream(self, _): pass


def test_recording_encodes_browser_mp4(tmp_path):
    manager = StreamManager(DummyRuntime(), default_fps=5)
    manager.recording_dir = tmp_path
    state = StreamState(stream_id="cam", source="x", target_fps=5)
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    now = time.time()
    state.latest_jpeg = buf.tobytes()
    state.replay_buffer = deque((now + i / 5, buf.tobytes()) for i in range(5))
    state.actual_fps = 5
    manager._states["cam"] = state
    info = manager.trigger_recording("cam", "42", post_seconds=1)
    clip_id = info["clip_id"]
    rec = state.recordings.pop(clip_id)
    manager._write_recording(clip_id, "cam", rec)
    status = manager.recording_status(clip_id)
    assert status["status"] == "ready"
    assert status["frame_count"] == 5
    assert status["duration_seconds"] == 1.0
    assert manager.recording_path(clip_id).read_bytes()[4:8] == b"ftyp"
