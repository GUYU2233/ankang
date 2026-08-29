#!/usr/bin/env python3
"""P4 KPI 评估：用微调后模型对躺卧场景视频抽帧，统计关键点检出。

业务 KPI（修正后）：
- outdoor + 低清躺卧：关键点检出 ≥ 8（肩/髋齐全）
- 站立帧召回不退化 ≥ 15 点
运行：ai-engine/.venv/bin/python scripts/eval_pose_ft_kpi.py --model <权重路径>
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ai-engine/runs/pose_ft/pilot_prelabel/weights/best.pt",
                    help="微调后权重")
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()

    model = YOLO(str(PROJECT_ROOT / args.model))

    # 1) 语料躺卧帧按场景统计（Pilot images）
    rows = list(csv.DictReader(open(PROJECT_ROOT / "data/annotations/pose_ft/manifest.csv",
                                    encoding="utf-8-sig")))
    lying = [r for r in rows if r["action_label"] == "lying"]
    print(f"=== 语料躺卧帧（{len(lying)} 张）分场景检出 ===")
    from collections import defaultdict
    scene_stats = defaultdict(lambda: [0, 0, 0.0])  # total, detected, kp_sum
    for r in lying:
        p = PROJECT_ROOT / "data/annotations/pose_ft/images" / r["train_val"] / r["filename"]
        if not p.is_file():
            continue
        res = model.predict(str(p), verbose=False, conf=args.conf, device="cpu")[0]
        n = 0 if res.keypoints is None else len(res.keypoints)
        k = int((res.keypoints.conf[0] >= 0.25).sum()) if n > 0 else 0
        s = scene_stats[r["scene"]]
        s[0] += 1
        s[1] += 1 if n > 0 else 0
        s[2] += k
    for sc, (tot, det, kp) in sorted(scene_stats.items()):
        print(f"  {sc:<12} {det}/{tot} 检出人 | 平均关键点 {kp/tot:.1f}")

    # 2) 卫生间视频（KPI 参考项：3D 模拟 UI 图，预期仍低）
    print("=== 卫生间视频躺卧段（参考项）===")
    import cv2
    bath = [r for r in csv.DictReader(open(PROJECT_ROOT / "data/manifest.csv",
                                           encoding="utf-8-sig"))
            if "bathroom" in r["video_path"] and r["action_label"] == "fall"]
    for r in bath:
        cap = cv2.VideoCapture(str(PROJECT_ROOT / "data" / r["video_path"]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        end = float(r["fall_end_ts"] or 0)
        for t in [end + 0.5, end + 1.5, end + 2.5]:
            t = min(t, dur - 0.5)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            res = model.predict(frame, verbose=False, conf=args.conf, device="cpu")[0]
            n = 0 if res.keypoints is None else len(res.keypoints)
            k = int((res.keypoints.conf[0] >= 0.25).sum()) if n > 0 else 0
            print(f"  {r['video_path']} t={t:.1f}s: person={n} kp={k}")
        cap.release()

    # 3) 站立帧（防退化检查）
    print("=== 站立帧防退化（抽查 20 张）===")
    normal = [r for r in rows if r["action_label"] == "normal"][:20]
    kps = []
    for r in normal:
        p = PROJECT_ROOT / "data/annotations/pose_ft/images" / r["train_val"] / r["filename"]
        if not p.is_file():
            continue
        res = model.predict(str(p), verbose=False, conf=0.25, device="cpu")[0]
        n = 0 if res.keypoints is None else len(res.keypoints)
        k = int((res.keypoints.conf[0] >= 0.25).sum()) if n > 0 else 0
        kps.append(k)
    if kps:
        print(f"  站立帧平均关键点: {sum(kps)/len(kps):.1f}（目标 ≥15）")


if __name__ == "__main__":
    main()
