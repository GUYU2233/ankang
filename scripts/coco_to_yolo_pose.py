#!/usr/bin/env python3
"""CVAT 导出的 COCO Keypoints 1.0 JSON → YOLO-pose txt（P4 转换脚本）。

运行解释器：/home/ljh/ankang/backend/.venv/bin/python（本脚本只依赖标准库，无需 cv2/numpy）。
项目根 = Path(__file__).resolve().parents[1]（即 /home/ljh/ankang），一切相对路径基于它，不依赖 cwd。

【CLI】
  --input      单个 json 或目录（目录时扫描 *.json 合并处理），必填
  --img-root   图片根目录，默认 data/annotations/pose_ft/images
  --out-root   输出根目录，默认 <img-root>/../labels
  --dry-run    只解析/校验/打印，不写任何文件

【逻辑】
  (1) 读 JSON：categories 必须恰含 1 个（或取 name=='person'），断言其 keypoints 长度 == 17
      （COCO 17 点顺序，见 COCO_KEYPOINT_NAMES，与 prelabel_standing.py / ANNOTATION_GUIDE.md 逐字一致）；
      每个 annotation 的 keypoints 长度必须 51，否则报错并指明 image_id。
  (2) bbox 为 CVAT 惯例 [x,y,w,h] 像素；图宽高取自 images[] 条目。
  (3) 归一化：cx=(x+w/2)/W、cy=(y+h/2)/H、w/W、h/H，clip 到 [0,1]；
      17 组 (x/W, y/H, v)，v 原样保留 0/1/2（v==0 时该点坐标写 0 0 0）；点坐标同样 clip 到 [0,1]。
  (4) 输出行格式：'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} x1 y1 v1 ... x17 y17 v17'（空格分隔，class=0）；
      每图可多行（多人）；CVAT 导出过但零标注的图写空 txt。
  (5) file_name 归位：依次尝试 img_root/train/<file_name>、img_root/val/<file_name>、
      img_root/<file_name>，据此定 split（train/val）；找不到则列出清单并 exit 1。
      （根目录兜底命中时记为 split=train 并打印警告清单——Pilot 语料全部图片都在 train/ 或 val/ 下，
       正常流程不会触发。）
  (6) 输出 out_root/{split}/{stem}.txt（stem 取 file_name 去扩展名）。

【校验与报告】
  - 打印每 split 的图片数 / 实例数 / 平均每实例已标关键点数（v>=1 计数）；零实例图仅警告。
  - 断言每个输出 txt 对应存在的 jpg；断言每行恰好 17 个三元组（字段数 1+4+51=56）。
  - --dry-run 只打印不写文件。

【多文件合并】--input 为目录时按文件名排序逐个读取，同一图片（按归位后的 (split, stem)）的
标注行按文件顺序追加合并，最后一次性写出——多个 CVAT 任务导出同一批图时可安全合并多人标注。
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMG_ROOT = PROJECT_ROOT / "data" / "annotations" / "pose_ft" / "images"

# 与 prelabel_standing.py / ANNOTATION_GUIDE.md 逐字一致（交叉核对点）。
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KEYPOINTS = 17          # COCO 17 点
KEYPOINTS_PER_ANN = NUM_KEYPOINTS * 3   # 51

# 输出行字段数：class + bbox 4 + 17 三元组 = 56
FIELDS_PER_LINE = 1 + 4 + KEYPOINTS_PER_ANN


def clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def load_coco(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("images") or [], data.get("annotations") or [], data.get("categories") or []


def pick_category(categories):
    """categories 必须恰含 1 个（或取 name=='person'），keypoints 长度必须 17。"""
    if not categories:
        raise ValueError("categories 为空：该 JSON 不是有效的 COCO Keypoints 导出")
    if len(categories) == 1:
        cat = categories[0]
    else:
        matched = [c for c in categories if c.get("name") == "person"]
        if len(matched) != 1:
            raise ValueError(f"categories 数量为 {len(categories)} 且 name=='person' 的条目数 != 1")
        cat = matched[0]
    kps = cat.get("keypoints") or []
    if len(kps) != NUM_KEYPOINTS:
        raise ValueError(f"categories[0]['keypoints'] 长度必须为 {NUM_KEYPOINTS}，实际 {len(kps)}")
    return cat


def locate_image(img_root: Path, file_name: str):
    """依次尝试 train/、val/、根目录；返回 (split, 实际文件路径) 或 None。"""
    for split in ("train", "val"):
        p = img_root / split / file_name
        if p.is_file():
            return split, p
    p = img_root / file_name
    if p.is_file():
        return None, p  # 根目录兜底，split 由调用方定（train）并警告
    return None


def render_line(cx, cy, w, h, points):
    """points: 17 个 (x, y, v)；输出一行 56 字段。"""
    tokens = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for px, py, v in points:
        tokens.append(f"{px:.6f}")
        tokens.append(f"{py:.6f}")
        tokens.append(str(v))
    return " ".join(tokens)


def process_file(path: Path, img_root: Path, out_items: dict,
                 missing: list, root_fallback: list, stats: dict):
    """处理单个 JSON：校验 + 归一化，结果累积进 out_items。

    out_items: {(split, stem): [line, ...]}——零标注图该 key 的列表保持为空（写空 txt）。
    多文件合并时同一 (split, stem) 的标注行按文件顺序追加。
    """
    images, annotations, categories = load_coco(path)
    pick_category(categories)  # 校验类别与 17 点顺序

    image_map_local = {}
    for img in images:
        fid = img.get("id")
        file_name = img.get("file_name")
        if not file_name:
            raise ValueError(f"images[] 条目 id={fid} 缺少 file_name")
        loc = locate_image(img_root, file_name)
        if loc is None:
            missing.append(file_name)
            image_map_local[fid] = None  # 文件缺失：annotation 循环里跳过，最终 exit 1
            continue
        split, _ = loc
        if split is None:
            split = "train"
            root_fallback.append(file_name)
        key = (split, Path(file_name).stem)
        out_items.setdefault(key, [])  # 零标注图也建 key → 空 txt
        W, H = float(img["width"]), float(img["height"])
        if W <= 0 or H <= 0:
            raise ValueError(f"image_id={fid} ({file_name}) width/height 非法: {W}x{H}")
        image_map_local[fid] = (img, Path(file_name).stem)

    ann_count = 0
    for ann in annotations:
        kp = ann.get("keypoints")
        if not kp or len(kp) != KEYPOINTS_PER_ANN:
            raise ValueError(
                f"annotation id={ann.get('id')} image_id={ann.get('image_id')} "
                f"keypoints 长度必须为 {KEYPOINTS_PER_ANN}，实际 {len(kp) if kp else 0}"
            )
        img_entry = image_map_local.get(ann.get("image_id"))
        if img_entry is None:
            if ann.get("image_id") in image_map_local:
                continue  # 该图文件在磁盘缺失（已计入 missing，最终 exit 1）
            raise ValueError(
                f"annotation id={ann.get('id')} 引用不存在的 image_id={ann.get('image_id')}"
            )
        img, stem = img_entry
        file_name = img["file_name"]
        loc = locate_image(img_root, file_name)
        if loc is None:
            missing.append(file_name)  # 已在上一循环报过，这里兜底
            continue
        split, _ = loc
        if split is None:
            split = "train"
        key = (split, stem)
        W, H = float(img["width"]), float(img["height"])
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            raise ValueError(
                f"annotation id={ann.get('id')} image_id={ann.get('image_id')} bbox 必须为 [x,y,w,h] 4 元素"
            )
        x, y, w, h = (float(v) for v in bbox)
        cx = clip01((x + w / 2.0) / W)
        cy = clip01((y + h / 2.0) / H)
        nw = clip01(w / W)
        nh = clip01(h / H)

        points = []
        for i in range(NUM_KEYPOINTS):
            px, py, v = kp[3 * i], kp[3 * i + 1], kp[3 * i + 2]
            v = int(round(v))
            if v not in (0, 1, 2):
                raise ValueError(
                    f"annotation id={ann.get('id')} image_id={ann.get('image_id')} "
                    f"关键点 {i} 的可见性 v={v} 非法（必须 0/1/2）"
                )
            if v == 0:
                points.append((0.0, 0.0, 0))
            else:
                points.append((clip01(float(px) / W), clip01(float(py) / H), v))

        out_items[key].append(render_line(cx, cy, nw, nh, points))
        ann_count += 1
        stats["instances"][split] += 1
        stats["kp_visible"][split] += sum(1 for _, _, v in points if v >= 1)

    return ann_count


def main():
    parser = argparse.ArgumentParser(
        description="CVAT 导出的 COCO Keypoints 1.0 JSON → YOLO-pose txt（P4 转换脚本）"
    )
    parser.add_argument("--input", required=True,
                        help="COCO Keypoints JSON 文件或目录（目录时扫描 *.json 合并处理）")
    parser.add_argument("--img-root", default=str(DEFAULT_IMG_ROOT),
                        help=f"图片根目录（默认 {DEFAULT_IMG_ROOT}）")
    parser.add_argument("--out-root", default=None,
                        help="输出根目录（默认 <img-root>/../labels）")
    parser.add_argument("--dry-run", action="store_true", help="只解析/校验/打印，不写任何文件")
    args = parser.parse_args()

    img_root = Path(args.img_root)
    if not img_root.is_dir():
        print(f"[错误] img-root 不存在: {img_root}", file=sys.stderr)
        sys.exit(1)
    out_root = Path(args.out_root) if args.out_root else img_root.parent / "labels"

    input_path = Path(args.input)
    if input_path.is_dir():
        json_files = sorted(p for p in input_path.glob("*.json") if p.is_file())
        if not json_files:
            print(f"[错误] --input 目录中没有 *.json: {input_path}", file=sys.stderr)
            sys.exit(1)
    elif input_path.is_file():
        json_files = [input_path]
    else:
        print(f"[错误] --input 不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    stats = {"instances": defaultdict(int), "kp_visible": defaultdict(int)}
    out_items = {}
    missing, root_fallback = [], []

    total_ann = 0
    for jf in json_files:
        try:
            n = process_file(jf, img_root, out_items, missing, root_fallback, stats)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[错误] 解析 {jf} 失败: {e}", file=sys.stderr)
            sys.exit(1)
        total_ann += n
        print(f"  读入 {jf.name}: {n} 条 annotation")

    # —— 校验 ——
    if missing:
        print(f"[错误] 以下 {len(missing)} 个 file_name 在图片根目录找不到（train/val/根 均无）:",
              file=sys.stderr)
        for name in sorted(set(missing)):
            print(f"  - {name}", file=sys.stderr)
        print("提示：确认 CVAT 导出任务与 data/annotations/pose_ft/images 是同一批图（同目录导入）。",
              file=sys.stderr)
        sys.exit(1)

    if root_fallback:
        print(f"[警告] {len(root_fallback)} 个 file_name 仅命中图片根目录（不在 train/val 下），"
              f"已按 train 处理：")
        for name in sorted(set(root_fallback)):
            print(f"  - {name}")

    # 断言：每行 17 个三元组 / 56 字段 / 首字段 0；每个输出 txt 对应存在的 jpg
    bad_lines = 0
    for (split, stem), lines in out_items.items():
        jpg_candidates = [img_root / split / f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png")]
        if not any(p.is_file() for p in jpg_candidates):
            print(f"[错误] 输出 txt 对应的 jpg 不存在: {split}/{stem} (检查过 {jpg_candidates[0]})",
                  file=sys.stderr)
            sys.exit(1)
        for ln in lines:
            fields = ln.split(" ")
            if len(fields) != FIELDS_PER_LINE or fields[0] != "0":
                print(f"[错误] {split}/{stem}.txt 行字段数 != {FIELDS_PER_LINE} 或首字段 != '0': "
                      f"{ln[:80]}", file=sys.stderr)
                bad_lines += 1
    if bad_lines:
        sys.exit(1)

    # —— 报告 ——
    img_count = defaultdict(int)
    for (split, _stem), lines in out_items.items():
        img_count[split] += 1  # 唯一图片数（按归位后的 key 去重，多文件合并不重复计数）
    zero_inst_images = [f"{s}/{st}" for (s, st), ln in out_items.items() if not ln]
    print("\n=== 转换报告 ===")
    for split in ("train", "val"):
        n_img = img_count.get(split, 0)
        n_inst = stats["instances"].get(split, 0)
        avg_kp = (stats["kp_visible"].get(split, 0) / n_inst) if n_inst else 0.0
        print(f"  {split:<5} 图片 {n_img:>4} | 实例 {n_inst:>4} | 平均每实例已标关键点 {avg_kp:.1f}")
    print(f"  合计 图片 {sum(img_count.values())} | 实例 {sum(stats['instances'].values())} | "
          f"annotation 行 {total_ann}")

    if zero_inst_images:
        shown = zero_inst_images[:10]
        more = len(zero_inst_images) - len(shown)
        print(f"[警告] {len(zero_inst_images)} 张图零标注（将写空 txt，作为背景参与训练）:")
        for s in shown:
            print(f"  - {s}")
        if more > 0:
            print(f"  ... 其余 {more} 张略")

    if args.dry_run:
        print("\n[DRY-RUN] 未写入任何文件（--dry-run）。")
        return

    # —— 写出 ——
    n_written = 0
    for (split, stem), lines in sorted(out_items.items()):
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{stem}.txt"
        content = ("\n".join(lines) + "\n") if lines else ""
        target.write_text(content, encoding="utf-8")
        n_written += 1
    print(f"\n已写出 {n_written} 个 txt → {out_root}")


if __name__ == "__main__":
    main()
