#!/usr/bin/env python3
"""用 yolo26s-pose 对躺卧帧批量推理，产出 CVAT 可导入的 COCO Keypoints 1.0 预标注 JSON。

复用 prelabel_standing.py 的 COCO 输出结构，仅选帧逻辑改为 action_label=='lying'。
运行解释器：/home/ljh/ankang/ai-engine/.venv/bin/python

【CLI】
  --model    默认 ai-engine/models/yolo26s-pose.pt
  --manifest 默认 data/annotations/pose_ft/manifest.csv
  --out      默认 data/annotations/pose_ft/prelabels/prelabel_lying.json
  --conf     默认 0.05（躺卧姿态基模置信度低，放宽阈值）
  --split    all|train|val，默认 all
  --limit    冒烟用，默认 0
  --device   默认 cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO  # noqa: E402

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_csv(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select_lying_frames(rows, split, limit, skip=0):
    """只选 action_label=='lying' 的帧。返回 [(row, img_path)]。"""
    selected = []
    for row in rows:
        if split != "all" and row.get("train_val") != split:
            continue
        if row.get("action_label", "") == "lying":
            selected.append(row)
    if skip:
        selected = selected[skip:]
    if limit:
        selected = selected[:limit]
    frames = []
    for row in selected:
        img_path = PROJECT_ROOT / "data" / "annotations" / "pose_ft" / "images" \
            / row["train_val"] / row["filename"]
        if not img_path.is_file():
            print(f"[警告] 帧文件不存在，跳过: {img_path}", file=sys.stderr)
            continue
        frames.append((row, img_path))
    return frames


def build_output(frames, results_iter):
    """按帧序消费推理结果，构造 CVAT COCO Keypoints 1.0 JSON（与 prelabel_standing 一致）。"""
    images, annotations = [], []
    n_with_person = 0
    scores = []
    for i, (row, _p) in enumerate(frames):
        try:
            res = next(results_iter)
        except StopIteration:
            raise RuntimeError(f"推理结果数量少于输入帧数（第 {i} 帧 {row['filename']} 无结果）")
        except Exception as e:
            raise RuntimeError(f"推理第 {i} 帧 {row['filename']} 失败: {e}") from e

        H, W = res.orig_shape
        image_id = len(images)
        images.append({
            "id": image_id,
            "file_name": f"{row['train_val']}/{row['filename']}",
            "width": int(W),
            "height": int(H),
        })

        if res.boxes is None or len(res.boxes) == 0:
            continue
        n_with_person += 1

        xyxy = res.boxes.xyxy.cpu().numpy()
        box_conf = res.boxes.conf.cpu().numpy()
        kp_xy = res.keypoints.xy.cpu().numpy()
        kp_conf = res.keypoints.conf.cpu().numpy()

        for d in range(len(res.boxes)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[d])
            kps = []
            for j in range(17):
                px, py = float(kp_xy[d, j, 0]), float(kp_xy[d, j, 1])
                c = float(kp_conf[d, j])
                v = 2 if c >= 0.5 else (1 if c >= 0.25 else 0)
                kps += [round(px, 2), round(py, 2), v]
            annotations.append({
                "id": len(annotations),
                "image_id": image_id,
                "category_id": 1,
                "keypoints": kps,
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "iscrowd": 0,
                "score": round(float(box_conf[d]), 4),
            })
            scores.append(float(box_conf[d]))

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "person",
            "keypoints": list(COCO_KEYPOINT_NAMES),
            "skeleton": [list(s) for s in COCO_SKELETON],
        }],
    }, n_with_person, scores


def main():
    parser = argparse.ArgumentParser(description="躺卧帧预标注 → CVAT COCO Keypoints 1.0")
    parser.add_argument("--model", default="ai-engine/models/yolo26s-pose.pt")
    parser.add_argument("--manifest", default="data/annotations/pose_ft/manifest.csv")
    parser.add_argument("--out", default="data/annotations/pose_ft/prelabels/prelabel_lying.json")
    parser.add_argument("--conf", type=float, default=0.05, help="person 框置信度阈值（躺卧基模低，放宽）")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0, help="跳过前 N 帧（分批跑用）")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_path = resolve(args.model)
    manifest_path = resolve(args.manifest)
    out_path = resolve(args.out)

    if not model_path.is_file():
        print(f"[错误] 模型权重不存在: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not manifest_path.is_file():
        print(f"[错误] manifest 不存在: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    if not (0 < args.conf < 1):
        print(f"[错误] --conf 必须在 (0,1) 内", file=sys.stderr)
        sys.exit(1)

    rows = load_csv(manifest_path)
    frames = select_lying_frames(rows, args.split, args.limit, args.skip)
    print(f"选中躺卧帧 {len(frames)} 张")
    if not frames:
        print("[错误] 选中帧为空", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    model = YOLO(str(model_path))
    paths = [str(p) for _, p in frames]
    results_iter = iter(model.predict(paths, stream=True, conf=args.conf,
                                      device=args.device, verbose=False))
    output, n_with_person, scores = build_output(frames, results_iter)
    elapsed = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    avg_conf = (sum(scores) / len(scores)) if scores else 0.0
    print(f"处理帧 {len(output['images'])} | 含人帧 {n_with_person} | "
          f"person 实例 {len(output['annotations'])} | 平均置信度 {avg_conf:.3f}")
    print(f"耗时 {elapsed:.1f}s | 输出 {out_path}")


if __name__ == "__main__":
    main()
