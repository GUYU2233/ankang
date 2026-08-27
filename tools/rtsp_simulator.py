#!/usr/bin/env python3
"""生成模拟居家监控视频，用于无真实摄像头时验证 AI 引擎与系统闭环。

用法:
    python tools/rtsp_simulator.py --out data/sim_videos/livingroom_fall.mp4 --seconds 30 --scene livingroom

Docker/真机环境如需 RTSP 推流（本机未安装 ffmpeg，部署环境可用）:
    ffmpeg -re -stream_loop -1 -i data/sim_videos/livingroom_fall.mp4 -c copy -f rtsp rtsp://zlm:8554/live/livingroom
"""

import argparse
import math
import os

import cv2
import numpy as np


def draw_frame(t: int, scene: str, fall_period: bool) -> np.ndarray:
    h, w = 720, 1280
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (52, 56, 68)
    cv2.rectangle(img, (0, int(h * 0.55)), (w, h), (88, 92, 104), -1)
    cv2.putText(img, f"SIM {scene.upper()} t={t // 25}s", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 210, 210), 2)

    cx = int(w * 0.5 + 120 * math.sin(t / 45.0))
    floor_y = int(h * 0.88)
    bh, bw = 340, 150
    angle = 6.0 * math.sin(t / 15.0)
    if fall_period:
        if t % 100 < 20:
            angle = 55.0
            bh, bw = 160, 320
        else:
            bh, bw = 100, 380
            cx = int(w * 0.52)
    pts = np.array([
        [cx - bw // 2, floor_y - bh // 3],
        [cx, floor_y - bh],
        [cx + bw // 2, floor_y - bh // 3],
        [cx + bw // 4, floor_y],
        [cx - bw // 4, floor_y],
    ], dtype=np.int32)
    M = cv2.getRotationMatrix2D((cx, floor_y), angle, 1.0)
    pts = cv2.transform(pts.reshape(1, -1, 2), M).reshape(-1, 2).astype(np.int32)
    cv2.fillConvexPoly(img, pts, (70, 170, 255))
    cv2.drawContours(img, [pts], 0, (0, 0, 0), 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sim_videos/sim_demo.mp4")
    ap.add_argument("--seconds", type=int, default=40)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--scene", default="livingroom")
    ap.add_argument("--with-fall", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 720))
    total = args.seconds * args.fps
    for t in range(total):
        frame = draw_frame(t, args.scene, fall_period=args.with_fall and t > args.fps * 2)
        writer.write(frame)
    writer.release()
    print(f"已生成模拟视频: {args.out}  ({args.seconds}s, {args.fps}fps)")


if __name__ == "__main__":
    main()