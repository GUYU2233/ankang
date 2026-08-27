#!/usr/bin/env python3
"""根据 docs/数据集要求.md 的目录结构，扫描 data/videos 生成 manifest.csv 并做基础校验。"""

import csv
import json
import os
from pathlib import Path

ROOT = Path("data")
VIDEO_EXTS = {".mp4", ".avi", ".mkv"}
CATEGORIES = ["fall", "nearfall", "normal", "risk_behavior"]
LABEL_BY_CATEGORY = {"fall": "fall", "nearfall": "nearfall", "normal": "normal", "risk_behavior": "risk_behavior"}


def main() -> None:
    videos_root = ROOT / "videos"
    anno_root = ROOT / "annotations" / "action_events"
    rows = []
    issues = []
    if not videos_root.exists():
        print("data/videos 不存在，请先按数据集要求放置视频")
        return

    for scene_dir in sorted(videos_root.iterdir()) if videos_root.is_dir() else []:
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name
        for cat in CATEGORIES:
            cat_dir = scene_dir / cat
            if not cat_dir.exists():
                continue
            for video in sorted(cat_dir.iterdir()):
                if video.suffix.lower() not in VIDEO_EXTS:
                    continue
                rel = video.relative_to(Path(".")).as_posix()
                anno_file = anno_root / (video.stem + ".json")
                fall_start, fall_end = "", ""
                if cat in ("fall", "nearfall") and not anno_file.exists():
                    issues.append(f"缺少标注: {rel}")
                if anno_file.exists():
                    try:
                        anno = json.loads(anno_file.read_text(encoding="utf-8"))
                        fall_start = anno.get("fall_start_ts", "")
                        fall_end = anno.get("fall_end_ts", "")
                    except Exception as exc:
                        issues.append(f"标注解析失败: {anno_file} ({exc})")
                rows.append([
                    rel, scene, cat, LABEL_BY_CATEGORY[cat],
                    "", "", "", "", "", "", "", "",
                    fall_start, fall_end,
                    anno_file.as_posix() if anno_file.exists() else "",
                    "", ""
                ])

    if not rows:
        print("未发现视频，请检查 data/videos/{scene}/{类别}/ 目录")
        return

    manifest = ROOT / "manifest.csv"
    header = ["video_path", "scene", "event_type", "action_label", "subject_id", "age_group", "gender",
              "lighting", "camera_angle", "occlusion", "resolution", "fps", "duration_s",
              "fall_start_ts", "fall_end_ts", "annotation_file", "sensor_file", "note"]
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"已生成 {manifest}，共 {len(rows)} 条记录")
    if issues:
        print(f"校验问题 {len(issues)} 条:")
        for x in issues[:20]:
            print(" -", x)


if __name__ == "__main__":
    main()