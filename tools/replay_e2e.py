"""端到端回放测试：本地视频 -> 15FPS 流式采样 -> 姿态 -> 时序 ONNX -> 跌倒判定。

用法：
    python tools/replay_e2e.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai-engine"))
os.environ.setdefault("TEMPORAL_MODEL_PATH", str(ROOT / "ai-engine" / "models" / "tcn_fall.onnx"))

from app.runtime import runtime
from app.streaming import StreamManager


def replay(source: str, stream_id: str, target_fps: float, timeout_s: float) -> dict:
    sm = StreamManager(runtime, default_fps=target_fps)
    sm.start(stream_id, source, target_fps, loop_file=False)
    max_fall_prob = 0.0
    fall_detected = False
    frames = 0
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        res = sm.latest(stream_id)
        if res:
            p = float(res.get("fall_prob", 0.0))
            max_fall_prob = max(max_fall_prob, p)
            if res.get("fall_detected"):
                fall_detected = True
        status = sm.status(stream_id)
        frames = status.get("frames_processed", frames)
        if not status.get("running") and sm.latest(stream_id) is not None:
            break
        time.sleep(0.05)
    status = sm.status(stream_id)
    sm.stop(stream_id)
    return {
        "source": source,
        "max_fall_prob": round(max_fall_prob, 3),
        "fall_detected": fall_detected,
        "actual_fps": status.get("actual_fps"),
        "frames": frames,
        "error": status.get("error"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fall-video", default=str(ROOT / "data" / "videos" / "livingroom" / "fall" / "livingroom_fall_001_20260827.mp4"))
    ap.add_argument("--normal-video", default=str(ROOT / "data" / "videos" / "livingroom" / "normal" / "livingroom_adl_001_20260827.mp4"))
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--timeout", type=float, default=40.0)
    args = ap.parse_args()

    fall_src = args.fall_video if Path(args.fall_video).exists() else str(ROOT / args.fall_video)
    normal_src = args.normal_video if Path(args.normal_video).exists() else str(ROOT / args.normal_video)

    fall = replay(fall_src, "replay-fall", args.fps, args.timeout)
    normal = replay(normal_src, "replay-normal", args.fps, args.timeout)

    print("== 端到端回放结果 ==")
    print("fall  :", fall)
    print("normal:", normal)

    ok = fall["max_fall_prob"] > normal["max_fall_prob"]
    print("\n比较:", "PASS (跌倒视频风险高于正常视频)" if ok else "CHECK (需人工复核)")
    if fall["error"]:
        print("WARN fall stream error:", fall["error"])


if __name__ == "__main__":
    main()
