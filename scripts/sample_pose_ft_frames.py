#!/usr/bin/env python3
"""从 fall 视频采样躺卧帧+过渡帧、从 normal 视频采样站立帧，生成 YOLO26-pose 微调 Pilot 语料。

设计要点（对应主指挥任务书 P0）：
- 固定解释器 /home/ljh/ankang/backend/.venv/bin/python（cv2 5.0.0 + numpy 2.5.2），
  项目根 = Path(__file__).resolve().parents[1]（即 /home/ljh/ankang），一切相对路径基于它，不依赖 cwd。
- 视频池：池A = fall 且 fall_start_ts/fall_end_ts 均可 float 解析（实测 286 个，二者同集）；
  池B = fall 且两时间戳皆无（实测 32 个：URFD cam1 30 + GMDCSA-24 卧室 2）；
  池C = action_label == 'normal'（154 个）。nearfall/risk_behavior 不参与。
- Pilot 分层选择（最大余额法，按 source_dataset 分层，比例配额每源±2）：
  nA=round(target_transition)=80、nB=round((target_lying-3*nA)/2)=5、nC=target_standing=40；
  --full 时三池全选。各源内先按 video_path 排序再用 random.Random(seed) 洗牌取头部，保证确定性。
- 每视频抽帧（random.Random(seed) 单实例贯穿全程）：
  池A 躺卧 3 帧：优先从真实躺卧窗 W=[fall_end_ts, min(fall_end_ts+5, duration_s)] 取
    （0.2/0.5/0.8 分位；|W| 过短时分位帧可能去重为 2 帧）；
    W 为空（fall_end_ts>=duration_s，实测 livingroom_fall_208/228/248 等）回退
    [0.6*duration_s, 0.9*duration_s] 取 3 帧（0.2/0.5/0.8 分位）。
    说明：任务书配额（Pilot ≈370 帧 = 250 躺卧 + 80 过渡 + 40 站立；--full ≈1362 帧 =
    286*4+32*2+154*1）按每池A视频 3 躺卧帧校准；manifest 实测绝大多数视频的
    fall_end_ts 紧贴视频末尾（|W| 远小于 2s），故统一取 3 帧以落在强制配额区间内。
  池A 过渡 1 帧：ts=clamp(fall_start_ts, 0, duration_s)。
  池B：[0.6*duration_s, 0.9*duration_s] 随机 2 帧（rng.uniform），action_label='lying'。
  池C：[0.2s, duration_s-0.2s] 随机 1 帧，action_label 保持 'normal'。
- 抽帧：cv2.VideoCapture 打不开 → skipped_sources 并跳过；fps=CAP_PROP_FPS（0 则回退
  manifest fps 列）；frame_idx=min(round(ts*fps), total_frames-1)，先 POS_FRAMES seek，
  失败再 POS_MSEC 重试一次，仍失败计 skipped_frames。JPEG 质量 95，保持原始分辨率绝不
  resize/放大（low_res：resolution 含 '240' 或 '320'）。文件名 {video_stem}_t{ts:.1f}.jpg，
  同视频内重名改用 t{ts:.2f}（590 个 stem 全局唯一）。
- train/val 划分（组级，防泄漏）：组键 = source_video；URFD 同事件 cam0/cam1 必须同侧，
  组键 = note 中 source= 值去掉尾部 cam\\\\d（如 'URFD fall-01-cam0' -> 'URFD fall-01'）。
  组级 85/15 按 source_dataset 分层（最大余额法，每源至少 1 组进 train）。
- 输出：data/annotations/pose_ft/images/{train,val}/*.jpg、manifest.csv（utf-8-sig，
  列 filename,source_video,frame_ts,scene,action_label,source_dataset,low_res,train_val；
  frame_ts 2 位小数）、pilot.txt（每行 images/{split}/{filename}）。
- 幂等：先 shutil.rmtree 清空 out/images/ 再重建，覆盖 manifest.csv 与 pilot.txt；
  绝不触碰 labels/、cvat/、prelabels/（人工标注产物）。
- 内置断言（任一失败 exit 1）：(1) 无泄漏（source_video 与 URFD 组键不跨 train/val）；
  (2) 成功采样的池A视频各 1 过渡帧+1-3 躺卧帧、池B 2 帧、池C 1 帧；
  (3) manifest 每行 images/{train_val}/{filename} 存在且与 images 文件集合完全相等；
  (4) 文件名全局唯一；(5) Pilot 配额 lying∈[225,275]、transition∈[72,88]、standing∈[38,42]。
"""

import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifest.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "annotations" / "pose_ft"

MANIFEST_COLUMNS = [
    "filename", "source_video", "frame_ts", "scene", "action_label",
    "source_dataset", "low_res", "train_val",
]


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def parse_source_dataset(note: str) -> str:
    """note 含 'source=' 时取其后第一个空格或 '/' 前的 token，否则 'unknown'。"""
    m = re.search(r"source=", note)
    if not m:
        return "unknown"
    rest = note[m.end():].lstrip()
    token = re.split(r"[ /]", rest, maxsplit=1)[0]
    return token if token else "unknown"


def parse_source_id(note: str) -> str:
    """note 中 source= 的完整取值（到第一个 ';' 为止），如 'URFD fall-01-cam0'。"""
    m = re.search(r"source=", note)
    if not m:
        return ""
    return note[m.end():].lstrip().split(";", 1)[0].strip()


def group_key_for(row: dict) -> str:
    """组键：source_video；URFD 同事件 cam0/cam1 源 id 去掉尾部 cam\\d 后同键。"""
    sid = parse_source_id(row["note"])
    if sid and re.search(r"cam\d+$", sid):
        return re.sub(r"cam\d+$", "", sid)
    return row["video_path"]


def is_low_res(resolution: str) -> bool:
    return "240" in resolution or "320" in resolution


def largest_remainder_alloc(counts: dict, n: int) -> dict:
    """最大余额法把 n 个名额按 counts 比例分配到各源；tie 按源名确定序。"""
    sources = sorted(counts)
    total = sum(counts.values())
    if total <= 0:
        return {}
    quotas = {s: n * counts[s] / total for s in sources}
    alloc = {s: int(q) for s, q in quotas.items()}
    remainder = n - sum(alloc.values())
    order = sorted(sources, key=lambda s: (-(quotas[s] - int(quotas[s])), s))
    i = 0
    while remainder > 0 and order:
        s = order[i % len(order)]
        if alloc[s] < counts[s]:
            alloc[s] += 1
            remainder -= 1
        i += 1
        if i > len(order) * 1000:
            break
    return alloc


def stratified_select(pool: list, n: int, rng: random.Random) -> list:
    """按 source_dataset 分层选 n 个：源内按 video_path 排序 -> rng 洗牌 -> 取头部。"""
    if n <= 0:
        return []
    by_src = defaultdict(list)
    for row in pool:
        by_src[parse_source_dataset(row["note"])].append(row)
    alloc = largest_remainder_alloc({s: len(v) for s, v in by_src.items()}, n)
    picked = []
    for src in sorted(alloc):
        videos = sorted(by_src[src], key=lambda r: r["video_path"])
        rng.shuffle(videos)
        picked.extend(videos[: alloc[src]])
    return picked


def load_manifest(path: Path) -> list:
    """严格 csv.DictReader 解析（utf-8-sig 带 BOM；note 列分号逗号混合）。"""
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_pools(rows: list):
    pool_a = [r for r in rows if r["action_label"] == "fall"
              and is_float(r["fall_start_ts"]) and is_float(r["fall_end_ts"])]
    pool_b = [r for r in rows if r["action_label"] == "fall"
              and not is_float(r["fall_start_ts"]) and not is_float(r["fall_end_ts"])]
    pool_c = [r for r in rows if r["action_label"] == "normal"]
    return pool_a, pool_b, pool_c


def lying_timestamps(fe: float, dur: float) -> list:
    """池A 躺卧采样时刻（3 个分位；真实窗为空则回退 [0.6dur,0.9dur]）。"""
    w_lo = max(fe, 0.0)
    w_hi = min(fe + 5.0, dur)
    if w_hi - w_lo > 1e-9:
        lo, hi = w_lo, w_hi
    else:
        lo, hi = 0.6 * dur, 0.9 * dur
    return [lo + q * (hi - lo) for q in (0.2, 0.5, 0.8)]


def resolve_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


class FrameSampler:
    """单个视频的抽帧执行器：POS_FRAMES seek，失败回退 POS_MSEC。"""

    def __init__(self, video_path: Path, manifest_fps: float, rng: random.Random):
        self.video_path = video_path
        self.manifest_fps = manifest_fps
        self.rng = rng
        self.cap = None
        self.total_frames = 0
        self.fps = 0.0

    def open(self) -> bool:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            cap.release()
            return False
        self.cap = cap
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = n if n and n > 0 else 0
        if not fps or fps <= 0:
            fps = self.manifest_fps
        self.fps = float(fps) if fps and fps > 0 else 0.0
        return True

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_frame(self, ts: float):
        """返回 (frame, actual_ts, frame_idx) 或 (None, None, None)。"""
        if self.cap is None:
            return None, None, None
        if self.fps > 0 and self.total_frames > 0:
            frame_idx = min(int(round(ts * self.fps)), self.total_frames - 1)
            if frame_idx < 0:
                frame_idx = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame, frame_idx / self.fps if self.fps > 0 else ts, frame_idx
            # 重试一次：POS_MSEC
            self.cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame, ts, frame_idx
            return None, None, None
        # 无帧数信息：直接 POS_MSEC
        self.cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, frame = self.cap.read()
        if ret and frame is not None:
            return frame, ts, -1
        return None, None, None


def pick_filename(stem: str, ts: float, used: set) -> str:
    """{video_stem}_t{ts:.1f}.jpg；同视频内 .1f 重名改用 .2f，仍重名追加序号。"""
    name = f"{stem}_t{ts:.1f}.jpg"
    if name not in used:
        return name
    name = f"{stem}_t{ts:.2f}.jpg"
    if name not in used:
        return name
    i = 0
    while True:
        name = f"{stem}_t{ts:.2f}_{i}.jpg"
        if name not in used:
            return name
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26-pose 微调 Pilot 语料采样（fall 躺卧/过渡 + normal 站立）")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="manifest.csv 路径（默认 data/manifest.csv）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录（默认 data/annotations/pose_ft）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="组级 val 比例（默认 0.15）")
    parser.add_argument("--full", action="store_true", help="全量模式：三池全选")
    parser.add_argument("--target-lying", type=int, default=250, help="Pilot 躺卧配额（默认 250）")
    parser.add_argument("--target-transition", type=int, default=80, help="Pilot 过渡配额（默认 80）")
    parser.add_argument("--target-standing", type=int, default=40, help="Pilot 站立配额（默认 40）")
    parser.add_argument("--limit", type=int, default=0, help="调试用：每池最多选 N 个视频（跳过配额断言）")
    args = parser.parse_args()

    manifest_path = resolve_path(PROJECT_ROOT, args.manifest)
    out_root = resolve_path(PROJECT_ROOT, args.out)
    images_root = out_root / "images"

    print(f"== 采样脚本 ==")
    print(f"manifest: {manifest_path}")
    print(f"out:      {out_root}")
    print(f"seed:     {args.seed}  val_ratio: {args.val_ratio}  full: {args.full}  limit: {args.limit}")

    # ---------- 解析 manifest ----------
    rows = load_manifest(manifest_path)
    if not rows:
        print("ERROR: manifest 为空", file=sys.stderr)
        sys.exit(1)
    print(f"manifest 行数: {len(rows)}")

    # ---------- 视频池 ----------
    pool_a, pool_b, pool_c = build_pools(rows)
    print(f"池A(fall+ts): {len(pool_a)}  池B(fall 无 ts): {len(pool_b)}  池C(normal): {len(pool_c)}")

    n_a = round(args.target_transition)
    n_b = round((args.target_lying - 3 * n_a) / 2)
    n_c = args.target_standing
    if args.full:
        n_a, n_b, n_c = len(pool_a), len(pool_b), len(pool_c)
    if args.limit > 0:
        n_a = min(n_a, args.limit)
        n_b = min(n_b, args.limit)
        n_c = min(n_c, args.limit)

    rng = random.Random(args.seed)
    sel_a = pool_a if args.full else stratified_select(pool_a, n_a, rng)
    sel_b = pool_b if args.full else stratified_select(pool_b, n_b, rng)
    sel_c = pool_c if args.full else stratified_select(pool_c, n_c, rng)
    print(f"选中: 池A {len(sel_a)}  池B {len(sel_b)}  池C {len(sel_c)}")

    # ---------- 组级 train/val 划分（防泄漏） ----------
    groups = {}
    for row in sel_a + sel_b + sel_c:
        groups.setdefault(group_key_for(row), []).append(row)
    by_src_groups = defaultdict(list)
    for gk, v in groups.items():
        by_src_groups[parse_source_dataset(v[0]["note"])].append(gk)

    val_budget = int(round(args.val_ratio * len(groups)))
    val_counts = {}
    for src, gks in by_src_groups.items():
        if len(gks) <= 1:
            val_counts[src] = 0
            continue
        q = args.val_ratio * len(gks)
        val_counts[src] = min(int(q), len(gks) - 1)
    remainder = val_budget - sum(val_counts.values())
    order = sorted(by_src_groups, key=lambda s: (-(args.val_ratio * len(by_src_groups[s]) - int(args.val_ratio * len(by_src_groups[s]))), s))
    i = 0
    while remainder > 0 and order:
        s = order[i % len(order)]
        if len(by_src_groups[s]) > 1 and val_counts[s] < len(by_src_groups[s]) - 1:
            val_counts[s] += 1
            remainder -= 1
        i += 1
        if i > 1000:
            break
    val_groups = set()
    for src, gks in by_src_groups.items():
        ordered = sorted(gks)
        rng.shuffle(ordered)
        val_groups.update(ordered[: val_counts[src]])
    train_groups = set(groups) - val_groups
    print(f"组数: {len(groups)}（val {len(val_groups)} / train {len(train_groups)}）")
    print(f"  val 按源: {dict(sorted(val_counts.items()))}")

    # ---------- 清理并重建输出 ----------
    for keep in ("labels", "cvat", "prelabels"):
        p = out_root / keep
        if p.exists():
            print(f"注意: 输出目录下存在 {keep}/，本次运行不触碰")
    if images_root.exists():
        shutil.rmtree(images_root)
    (images_root / "train").mkdir(parents=True, exist_ok=True)
    (images_root / "val").mkdir(parents=True, exist_ok=True)

    # ---------- 抽帧 ----------
    manifest_rows = []
    used_names = set()
    per_video_frames = defaultdict(Counter)  # source_video -> {action_label: n}
    skipped_sources = []
    skipped_frames = 0
    frame_counters = Counter()  # action_label
    low_res_count = 0
    source_split = Counter()  # (source_dataset, train_val)

    def process_frame(sampler: FrameSampler, row: dict, ts: float, action_label: str,
                      split: str, stem: str) -> bool:
        nonlocal low_res_count
        frame, _, _ = sampler.read_frame(ts)
        if frame is None:
            return False
        name = pick_filename(stem, ts, used_names)
        used_names.add(name)
        rel = f"images/{split}/{name}"
        dst = out_root / rel
        ok = cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            print(f"WARN: 写入失败 {dst}", file=sys.stderr)
            return False
        low = is_low_res(row["resolution"])
        manifest_rows.append({
            "filename": name,
            "source_video": row["video_path"],
            "frame_ts": f"{ts:.2f}",
            "scene": row["scene"],
            "action_label": action_label,
            "source_dataset": parse_source_dataset(row["note"]),
            "low_res": "true" if low else "false",
            "train_val": split,
        })
        frame_counters[action_label] += 1
        per_video_frames[row["video_path"]][action_label] += 1
        if low:
            low_res_count += 1
        source_split[(parse_source_dataset(row["note"]), split)] += 1
        return True

    def frame_idx_of(sampler: FrameSampler, ts: float) -> int:
        if sampler.fps > 0 and sampler.total_frames > 0:
            return min(int(round(ts * sampler.fps)), sampler.total_frames - 1)
        return -1

    def sample_video(row: dict, split: str, requests: list) -> None:
        """requests: list[(ts, action_label)]。同标签请求按物理帧去重；不同标签不互相去重。"""
        nonlocal skipped_frames
        stem = Path(row["video_path"]).stem
        video_path = PROJECT_ROOT / "data" / row["video_path"]
        fps_manifest = float(row["fps"]) if is_float(row["fps"]) else 0.0
        sampler = FrameSampler(video_path, fps_manifest, rng)
        if not sampler.open():
            skipped_sources.append(row["video_path"])
            return
        done_by_label = defaultdict(set)
        for ts, action_label in requests:
            idx = frame_idx_of(sampler, ts)
            if idx in done_by_label[action_label]:
                continue  # 同标签同一物理帧去重
            if process_frame(sampler, row, ts, action_label, split, stem):
                done_by_label[action_label].add(idx)
            else:
                skipped_frames += 1
        sampler.close()

    # 先处理池A（躺卧+过渡），再池B，再池C
    for row in sel_a:
        split = "val" if group_key_for(row) in val_groups else "train"
        fe = float(row["fall_end_ts"])
        dur = float(row["duration_s"])
        fs = float(row["fall_start_ts"])
        trans_ts = min(max(fs, 0.0), dur)
        requests = [(t, "lying") for t in lying_timestamps(fe, dur)]
        requests.append((trans_ts, "transition"))
        sample_video(row, split, requests)
    for row in sel_b:
        split = "val" if group_key_for(row) in val_groups else "train"
        dur = float(row["duration_s"])
        lo, hi = 0.6 * dur, 0.9 * dur
        requests = [(rng.uniform(lo, hi), "lying") for _ in range(2)]
        sample_video(row, split, requests)
    for row in sel_c:
        split = "val" if group_key_for(row) in val_groups else "train"
        dur = float(row["duration_s"])
        lo, hi = 0.2, dur - 0.2
        if hi <= lo:
            ts = dur / 2.0
        else:
            ts = rng.uniform(lo, hi)
        sample_video(row, split, [(ts, "normal")])

    # ---------- 写 manifest.csv / pilot.txt ----------
    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (out_root / "pilot.txt").open("w", encoding="utf-8") as f:
        for m in manifest_rows:
            f.write(f"images/{m['train_val']}/{m['filename']}\n")

    # ---------- 统计 ----------
    print("\n== 帧统计 ==")
    print(f"lying: {frame_counters['lying']}  transition: {frame_counters['transition']}  normal(站立): {frame_counters['normal']}")
    print(f"总帧数: {sum(frame_counters.values())}")
    print(f"low_res 帧: {low_res_count}")
    print(f"train/val 帧: {sum(1 for m in manifest_rows if m['train_val']=='train')}/{sum(1 for m in manifest_rows if m['train_val']=='val')}")
    print("按 source_dataset × train/val:")
    for (src, split) in sorted(source_split):
        print(f"  {src:<15} {split}: {source_split[(src, split)]}")
    print("\n== 跳过 ==")
    print(f"skipped_sources: {len(skipped_sources)}")
    for v in skipped_sources:
        print(f"  - {v}")
    print(f"skipped_frames: {skipped_frames}")

    # ---------- 内置断言 ----------
    failures = []

    def check(cond: bool, msg: str):
        if cond:
            print(f"[PASS] {msg}")
        else:
            failures.append(msg)
            print(f"[FAIL] {msg}")

    # (1) 无泄漏
    train_vids = {m["source_video"] for m in manifest_rows if m["train_val"] == "train"}
    val_vids = {m["source_video"] for m in manifest_rows if m["train_val"] == "val"}
    leak_vids = train_vids & val_vids
    check(not leak_vids, f"无泄漏: source_video 不跨 train/val（交集 {len(leak_vids)}）")
    train_keys = {group_key_for(row) for row in sel_a + sel_b + sel_c
                  if group_key_for(row) in train_groups}
    val_keys = {group_key_for(row) for row in sel_a + sel_b + sel_c
                if group_key_for(row) in val_groups}
    leak_keys = train_keys & val_keys
    check(not leak_keys, f"无泄漏: URFD 组键不跨 train/val（交集 {len(leak_keys)}）")

    # (2) 每视频帧数
    per_ok = True
    for row in sel_a:
        sv = row["video_path"]
        if sv in skipped_sources:
            continue
        cnt = per_video_frames.get(sv, Counter())
        lying_n = cnt.get("lying", 0)
        trans_n = cnt.get("transition", 0)
        if not (1 <= lying_n <= 3) or trans_n != 1:
            per_ok = False
            print(f"  异常: 池A {sv} 躺卧={lying_n} 过渡={trans_n}")
    for row in sel_b:
        sv = row["video_path"]
        if sv in skipped_sources:
            continue
        cnt = per_video_frames.get(sv, Counter())
        if cnt.get("lying", 0) != 2:
            per_ok = False
            print(f"  异常: 池B {sv} 帧数 {cnt.get('lying', 0)}")
    for row in sel_c:
        sv = row["video_path"]
        if sv in skipped_sources:
            continue
        cnt = per_video_frames.get(sv, Counter())
        if cnt.get("normal", 0) != 1:
            per_ok = False
            print(f"  异常: 池C {sv} 帧数 {cnt.get('normal', 0)}")
    check(per_ok, "每视频帧数: 池A 1过渡+1-3躺卧 / 池B 2 / 池C 1")

    # (3) 文件存在且一一对应
    manifest_files = {m["filename"] for m in manifest_rows}
    on_disk = set()
    for split in ("train", "val"):
        for p in (images_root / split).iterdir():
            if p.is_file():
                on_disk.add(p.name)
    check(manifest_files == on_disk,
          f"文件集合一致: manifest {len(manifest_files)} == images {len(on_disk)}")
    missing = []
    for m in manifest_rows:
        p = out_root / "images" / m["train_val"] / m["filename"]
        if not p.exists():
            missing.append(str(p))
    check(not missing, f"manifest 每行文件存在（缺失 {len(missing)}）")

    # (4) 文件名全局唯一
    check(len(manifest_files) == len(manifest_rows), f"文件名全局唯一（{len(manifest_files)} 唯一名）")

    # (5) Pilot 配额（--full 不检查 Pilot 配额；--limit 调试模式跳过）
    if not args.full and args.limit <= 0:
        lying_n = frame_counters["lying"]
        trans_n = frame_counters["transition"]
        stand_n = frame_counters["normal"]
        check(225 <= lying_n <= 275, f"Pilot 躺卧配额: {lying_n} ∈ [225,275]")
        check(72 <= trans_n <= 88, f"Pilot 过渡配额: {trans_n} ∈ [72,88]")
        check(38 <= stand_n <= 42, f"Pilot 站立配额: {stand_n} ∈ [38,42]")
        total = sum(frame_counters.values())
        check(350 <= total <= 390, f"Pilot 总帧数: {total} ∈ [350,390]")
    if args.full:
        total = sum(frame_counters.values())
        check(total >= 1300, f"--full 总帧数: {total} >= 1300")

    # ---------- labels/ 孤儿警告 ----------
    labels_dir = out_root / "labels"
    if labels_dir.exists():
        orphan = []
        for p in labels_dir.rglob("*.txt"):
            if p.name not in {m["filename"].rsplit(".", 1)[0] + ".txt" for m in manifest_rows}:
                orphan.append(str(p))
        if orphan:
            print(f"\n警告: labels/ 下 {len(orphan)} 个无对应图像的孤儿文件（人工标注产物，未删除）:")
            for o in orphan[:10]:
                print(f"  - {o}")

    print("\n== 结果 ==")
    if failures:
        print(f"断言失败 {len(failures)} 项，exit 1")
        sys.exit(1)
    print("全部内置断言通过，exit 0")


if __name__ == "__main__":
    main()
