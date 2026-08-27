#!/usr/bin/env python3
"""数据完整性测试：读取 data/manifest.csv，校验文件存在性、标注字段、分辨率、时长与数量分布。

用法：
    python tools/test_dataset.py
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

DATA = Path("data")
MANIFEST = DATA / "manifest.csv"


def main() -> None:
    if not MANIFEST.exists():
        print("缺少 data/manifest.csv")
        return

    with MANIFEST.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"总记录数: {len(rows)}")

    # 1. 场景/类别分布
    dist = Counter((r["scene"], r["action_label"]) for r in rows)
    print("\n== 分布 scene/action_label ==")
    for (scene, label), n in sorted(dist.items()):
        print(f"  {scene:12s} {label:16s} {n}")

    # 2. 文件存在性
    missing_video = []
    missing_anno = []
    missing_sensor = []
    for r in rows:
        if not (DATA / r["video_path"]).exists():
            missing_video.append(r["video_path"])
        if r.get("annotation_file") and not (DATA / r["annotation_file"]).exists():
            missing_anno.append(r["annotation_file"])
        if r.get("sensor_file") and not (DATA / r["sensor_file"]).exists():
            missing_sensor.append(r["sensor_file"])
    print(f"\n== 文件缺失 ==")
    print(f"  video 缺失: {len(missing_video)}")
    print(f"  annotation 缺失: {len(missing_anno)}")
    print(f"  sensor 缺失: {len(missing_sensor)}")
    if missing_video[:5]:
        print("  video sample:", missing_video[:5])
    if missing_anno[:5]:
        print("  anno sample:", missing_anno[:5])
    if missing_sensor[:5]:
        print("  sensor sample:", missing_sensor[:5])

    # 3. 关键字段完整性
    fall_rows = [r for r in rows if r["action_label"] in ("fall", "nearfall")]
    no_ts = [r for r in fall_rows if not (r.get("fall_start_ts") and r.get("fall_end_ts"))]
    no_res = [r for r in rows if not r.get("resolution")]
    no_fps = [r for r in rows if not r.get("fps")]
    no_dur = [r for r in rows if not r.get("duration_s")]
    print(f"\n== 字段完整性 ==")
    print(f"  fall/nearfall 共 {len(fall_rows)}，缺少起止时间 {len(no_ts)}")
    print(f"  缺少 resolution: {len(no_res)}")
    print(f"  缺少 fps: {len(no_fps)}")
    print(f"  缺少 duration_s: {len(no_dur)}")

    # 4. 抽样解码
    print("\n== 抽样视频解码检查 ==")
    sample = rows[:5] + [r for r in rows if r["action_label"] == "fall"][:5]
    seen = set()
    for r in sample:
        key = r["video_path"]
        if key in seen:
            continue
        seen.add(key)
        cap = cv2.VideoCapture(str(DATA / key))
        if not cap.isOpened():
            print(f"  无法打开: {key}")
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = frames / fps if fps else 0
        ok, frame = cap.read()
        cap.release()
        print(f"  {key}: {w}x{h}, fps={fps:.2f}, frames={frames}, dur={dur:.2f}s, first_frame={'ok' if ok else 'FAIL'}")

    # 5. 统计总时长
    total_dur = 0.0
    for r in rows:
        try:
            total_dur += float(r.get("duration_s") or 0)
        except ValueError:
            pass
    print(f"\n总时长: {total_dur/60:.2f} 分钟 ({total_dur:.1f}s)")
    print("测试完成")


if __name__ == "__main__":
    main()