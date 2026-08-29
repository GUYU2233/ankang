#!/usr/bin/env python3
"""用 yolo26s-pose 对站立帧/过渡帧批量推理，产出 CVAT 可直接导入的 COCO Keypoints 1.0
预标注 JSON（P4 预标注脚本，备用方案，不依赖 nuclio）。

运行解释器：/home/ljh/ankang/ai-engine/.venv/bin/python（ultralytics 8.4.131 + torch 2.13.0+cpu）。
项目根 = Path(__file__).resolve().parents[1]（即 /home/ljh/ankang），一切相对路径基于它，不依赖 cwd。

【CLI】
  --model         预训练权重，默认 ai-engine/models/yolo26s-pose.pt
  --manifest      Pilot 帧清单，默认 data/annotations/pose_ft/manifest.csv
  --data-manifest 全量视频清单，默认 data/manifest.csv（用于查 fall_start_ts）
  --out           输出 JSON，默认 data/annotations/pose_ft/prelabels/prelabel.json
  --conf          person 框置信度阈值，默认 0.25
  --split         all|train|val，默认 all
  --limit         冒烟用：最多处理前 N 个选中帧（0=不限制）
  --device        默认 cpu

【选帧逻辑】
  standing  = pose_ft manifest 的 action_label=='normal'；
  transition = action_label=='transition'（'fall' 兼容）且 |frame_ts − fall_start_ts|≤0.6
    （fall_start_ts 由 source_video 关联 data/manifest.csv；躺卧帧 ts≈fall_end_ts 与
     fall_start 接近的边界重叠无害——实测躺卧帧距 fall_start 均 >0.6s，不会误入选）。
  躺卧帧刻意不预标注：基模对躺卧姿态弱（这正是微调要解决的），劣质预框会带来锚定偏差。
  Pilot 语料实测选中数 = 站立 40 + 过渡 80（±0）。

【推理】model.predict(图片路径列表, stream=True, conf=conf, device=device, verbose=False)。

【输出 JSON（CVAT COCO Keypoints 1.0）】
  images[]:      {id(顺序 0 起), file_name='{train|val}/{filename}'（相对 images 根）,
                  width, height}（尺寸取自 result.orig_shape）
  annotations[]: {id 顺序, image_id, category_id=1,
                  keypoints=[x,y,v]×17 像素坐标, bbox=[x,y,w,h] 像素（boxes.xyxy 换算）,
                  iscrowd=0, score=person 置信度}
  keypoints v 映射：该点置信 ≥0.5→2、[0.25,0.5)→1、<0.25→0（坐标仍写预测值）。
  多人时每检出一个人一条 annotation（标注员只留主体）；
  无检出帧保留在 images[]（CVAT 会显示该图）但无 annotation。
  categories[]: {id:1, name:'person', keypoints:[17 个 COCO 名称]（与转换脚本同一数组，
                 见 COCO_KEYPOINT_NAMES）, skeleton:COCO 标准}。
  按 manifest 顺序处理，输出确定。
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# —— 必须先于 ultralytics import（与 finetune_pose.py 一致的配置目录行为）——
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO  # noqa: E402

# 与 coco_to_yolo_pose.py / ANNOTATION_GUIDE.md 逐字一致（交叉核对点）。
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

FALL_WINDOW = 0.6  # 过渡窗：|frame_ts − fall_start_ts| ≤ 0.6s


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_csv(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select_frames(rows, fts_map, split, limit):
    """按 manifest 顺序选站立/过渡帧。返回 [(row, kind, img_path)]。"""
    selected = []
    for row in rows:
        if split != "all" and row.get("train_val") != split:
            continue
        label = row.get("action_label", "")
        if label == "normal":
            selected.append((row, "standing"))
        elif label in ("transition", "fall"):
            fs = fts_map.get(row.get("source_video", ""))
            if fs is not None and abs(float(row["frame_ts"]) - fs) <= FALL_WINDOW:
                selected.append((row, "transition"))
        # 其余（lying 等）刻意不预标注
    if limit:
        selected = selected[:limit]
    frames = []
    for row, kind in selected:
        img_path = Path(PROJECT_ROOT) / "data" / "annotations" / "pose_ft" / "images" \
            / row["train_val"] / row["filename"]
        if not img_path.is_file():
            print(f"[警告] 帧文件不存在，跳过: {img_path}", file=sys.stderr)
            continue
        frames.append((row, kind, img_path))
    return frames


def build_output(frames, results_iter, conf):
    """按帧序消费推理结果，构造 CVAT COCO Keypoints 1.0 JSON。"""
    images, annotations = [], []
    n_with_person = 0
    scores = []
    for i, (row, _kind, _p) in enumerate(frames):
        try:
            res = next(results_iter)
        except StopIteration:
            raise RuntimeError(f"推理结果数量少于输入帧数（第 {i} 帧 {row['filename']} 无结果）")
        except Exception as e:
            raise RuntimeError(f"推理第 {i} 帧 {row['filename']} 失败: {e}") from e

        H, W = res.orig_shape  # (height, width) 原图尺寸
        image_id = len(images)
        images.append({
            "id": image_id,
            "file_name": f"{row['train_val']}/{row['filename']}",
            "width": int(W),
            "height": int(H),
        })

        if res.boxes is None or len(res.boxes) == 0:
            continue  # 无检出：仅保留 images[] 条目
        n_with_person += 1

        xyxy = res.boxes.xyxy.cpu().numpy()      # (N,4)
        box_conf = res.boxes.conf.cpu().numpy()  # (N,)
        kp_xy = res.keypoints.xy.cpu().numpy()   # (N,17,2) 原图像素坐标
        kp_conf = res.keypoints.conf.cpu().numpy()  # (N,17)

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
    parser = argparse.ArgumentParser(
        description="用 yolo26s-pose 预标注站立/过渡帧 → CVAT COCO Keypoints 1.0 JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="ai-engine/models/yolo26s-pose.pt",
                        help="预训练权重（相对项目根解析）")
    parser.add_argument("--manifest", default="data/annotations/pose_ft/manifest.csv",
                        help="Pilot 帧清单（相对项目根解析）")
    parser.add_argument("--data-manifest", default="data/manifest.csv",
                        help="全量视频清单（用于查 fall_start_ts，相对项目根解析）")
    parser.add_argument("--out", default="data/annotations/pose_ft/prelabels/prelabel.json",
                        help="输出 JSON 路径（相对项目根解析）")
    parser.add_argument("--conf", type=float, default=0.25, help="person 框置信度阈值")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all",
                        help="处理哪个 split")
    parser.add_argument("--limit", type=int, default=0, help="冒烟用：最多处理前 N 个选中帧（0=不限）")
    parser.add_argument("--device", default="cpu", help="推理设备（默认 cpu）")
    args = parser.parse_args()

    model_path = resolve(args.model)
    manifest_path = resolve(args.manifest)
    data_manifest_path = resolve(args.data_manifest)
    out_path = resolve(args.out)

    if not model_path.is_file():
        print(f"[错误] 模型权重不存在: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not manifest_path.is_file() or not data_manifest_path.is_file():
        print(f"[错误] manifest 不存在: {manifest_path} 或 {data_manifest_path}", file=sys.stderr)
        sys.exit(1)
    if not (0 < args.conf < 1):
        print(f"[错误] --conf 必须在 (0,1) 内，实际 {args.conf}", file=sys.stderr)
        sys.exit(1)

    # fall_start_ts 查表（data/manifest.csv 为 utf-8-sig，video_path 相对 data/）
    fts_map = {}
    for r in load_csv(data_manifest_path):
        v = r.get("fall_start_ts", "")
        try:
            fts_map[r["video_path"]] = float(v)
        except (TypeError, ValueError):
            fts_map[r["video_path"]] = None

    rows = load_csv(manifest_path)
    frames = select_frames(rows, fts_map, args.split, args.limit)
    kind_counts = Counter(k for _, k, _ in frames)
    print(f"选中帧 {len(frames)}（站立 {kind_counts.get('standing', 0)} + "
          f"过渡 {kind_counts.get('transition', 0)}），躺卧帧不预标注")

    if not frames:
        print("[错误] 选中帧为空：请检查 manifest/--split 参数。", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    model = YOLO(str(model_path))
    paths = [str(p) for _, _, p in frames]
    results_iter = iter(model.predict(paths, stream=True, conf=args.conf,
                                      device=args.device, verbose=False))
    output, n_with_person, scores = build_output(frames, results_iter, args.conf)
    elapsed = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    avg_conf = (sum(scores) / len(scores)) if scores else 0.0
    print(f"处理帧 {len(output['images'])} | 含人帧 {n_with_person} | "
          f"person 实例 {len(output['annotations'])} | 平均置信度 {avg_conf:.3f}")
    print(f"耗时 {elapsed:.1f}s | 输出 {out_path}（json.load 可读，categories[0].keypoints 长度 "
          f"{len(output['categories'][0]['keypoints'])}）")


if __name__ == "__main__":
    main()
