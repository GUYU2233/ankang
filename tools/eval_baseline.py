#!/usr/bin/env python3
"""基线评测：用 ai-engine 的真实姿态启发式模型在 data/manifest.csv 抽样视频上做 clip 级预测。

注意：这是开发早期基线，不是最终识别结果；目标验证数据可读、链路可跑、模型有响应。
用法：
    python tools/eval_baseline.py --fall 8 --nearfall 6 --normal 6 --frames 6
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path("ai-engine").resolve()))
from app.pipelines.real_runtime import RealRuntime  # noqa: E402

DATA = Path("data")
MANIFEST = DATA / "manifest.csv"


def read_rows() -> list[dict]:
    with MANIFEST.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sample_videos(rows, label, n):
    chosen = [r for r in rows if r["action_label"] == label]
    step = max(1, len(chosen) // n) if n else 1
    return chosen[::step][:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fall", type=int, default=8)
    ap.add_argument("--nearfall", type=int, default=6)
    ap.add_argument("--normal", type=int, default=6)
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()

    rows = read_rows()
    real = RealRuntime()

    selected = []
    for label, n in [("fall", args.fall), ("nearfall", args.nearfall), ("normal", args.normal)]:
        selected.extend([(label, r) for r in sample_videos(rows, label, n)])

    print(f"共抽样 {len(selected)} 个视频，每段采样 {args.frames} 帧")
    print("label  video  pred  confirmed  max_risk  max_fall_prob  detections")
    per_label = defaultdict(lambda: {"total": 0, "pred_fall_loose": 0, "pred_fall_confirmed": 0})

    for label, r in selected:
        video_path = DATA / r["video_path"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"无法打开 {video_path}")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        indices = sorted(set(int(i * (frame_count - 1) / max(1, args.frames - 1)) for i in range(args.frames)))
        max_risk = 0.0
        max_fall_prob = 0.0
        detections = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            result = real.infer(frame)
            if not result:
                continue
            max_risk = max(max_risk, float(result.get("risk_score") or 0))
            max_fall_prob = max(max_fall_prob, float(result.get("fall_prob") or 0))
            if result.get("fall_detected"):
                detections += 1
        cap.release()

        pred = "fall" if (max_fall_prob >= 0.55 or detections > 0) else "nonfall"
        confirmed = "fall" if detections >= 2 else "nonfall"
        per_label[label]["total"] += 1
        per_label[label]["pred_fall_loose"] += 1 if pred == "fall" else 0
        per_label[label]["pred_fall_confirmed"] += 1 if confirmed == "fall" else 0
        print(f"{label:8s} {video_path.name:36s} {pred:7s} {confirmed:9s} {max_risk:.2f} {max_fall_prob:.2f} {detections}/{args.frames}")

    print("\n== 汇总 ==")
    for label in ("fall", "nearfall", "normal"):
        info = per_label[label]
        print(f"  {label:8s} loose_fall={info['pred_fall_loose']}/{info['total']} confirmed_fall={info['pred_fall_confirmed']}/{info['total']}")


if __name__ == "__main__":
    main()