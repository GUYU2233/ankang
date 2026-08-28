"""数据清单加载与按受试者划分。

说明：优先使用 manifest.csv 中明确的 subject_id；仅当 subject_id 缺失或为 unknown 时，
才用 source_file 派生受试者键，避免同一受试者/序列在训练集与验证集之间泄漏：

- URFD：fall-01-cam0 与 fall-01-cam1 是同一跌倒事件的两个机位，归入同一 subject key；
- Le2i：每个 source_file 是一段独立序列，各自作为独立 subject。
"""
from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path

_URFD_SEQ = re.compile(r"(fall|adl)-([0-9]+)", re.IGNORECASE)

ACTION_TO_BINARY = {"fall": 1, "nearfall": 0, "normal": 0, "risk_behavior": 0}
ACTION_TO_TERNARY = {"fall": 0, "nearfall": 1, "normal": 2, "risk_behavior": 2}
# 4 类：风险行为不再并入 normal / non_fall。
ACTION_TO_MULTICLASS = {"fall": 0, "nearfall": 1, "risk_behavior": 2, "normal": 3}


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_manifest(path: Path) -> list[dict]:
    return load_csv(path)


def load_source_mapping(path: Path | None) -> dict[str, dict]:
    if not path or not Path(path).exists():
        return {}
    mapping: dict[str, dict] = {}
    for row in load_csv(Path(path)):
        video = (row.get("new_video") or "").strip()
        if video:
            mapping[video] = row
    return mapping


def subject_key_from_source(source_dataset: str, source_file: str) -> str:
    """把来源文件映射为稳定的受试者/序列键。"""
    src = (source_dataset or "").strip().lower()
    f = (source_file or "").strip()
    if src == "urfd":
        m = _URFD_SEQ.search(f)
        if m:
            return f"urfd:{m.group(1)}-{int(m.group(2)):02d}"
        return "urfd:" + Path(f).stem
    if src == "le2i":
        clean = f.replace("\\", "/").rsplit(".", 1)[0]
        return "le2i:" + clean
    return f"{src or 'unknown'}:{Path(f).stem}"


def derive_subject_key(row: dict, mapping: dict[str, dict] | None = None) -> str:
    """从 manifest 行派生受试者键。优先明确 subject_id，再回退到来源序列。"""
    subject_id = (row.get("subject_id") or "").strip()
    if subject_id and subject_id.lower() not in {"unknown", "none", "null", "na", "n/a"}:
        return "subject:" + subject_id.lower()

    video = Path(row.get("video_path", "")).name
    m = (mapping or {}).get(video)
    if m and (m.get("source_dataset") or m.get("source_file")):
        return subject_key_from_source(m.get("source_dataset", ""), m.get("source_file", ""))

    note = row.get("note", "") or ""
    src_m = re.search(r"source=([A-Za-z0-9]+)\s+([A-Za-z0-9_./\\() -]+?)(?:;|$)", note)
    if src_m:
        return subject_key_from_source(src_m.group(1), src_m.group(2).strip())
    return "unknown:" + Path(video).stem


def annotate_subject_keys(rows: list[dict], mapping: dict[str, dict] | None = None) -> list[dict]:
    for r in rows:
        r["_subject_key"] = derive_subject_key(r, mapping)
    return rows


def label_from_row(row: dict, mode: str = "binary") -> int:
    action = (row.get("action_label") or "normal").strip().lower()
    if mode == "ternary":
        return ACTION_TO_TERNARY.get(action, 2)
    if mode == "multiclass":
        return ACTION_TO_MULTICLASS.get(action, 3)
    return ACTION_TO_BINARY.get(action, 0)


def label_names(mode: str = "binary") -> list[str]:
    if mode == "ternary":
        return ["fall", "nearfall", "normal"]
    if mode == "multiclass":
        return ["fall", "nearfall", "risk_behavior", "normal"]
    return ["non_fall", "fall"]


def split_by_subject(
    rows: list[dict],
    val_split: float = 0.2,
    seed: int = 42,
    classes: str = "binary",
) -> tuple[list[dict], list[dict]]:
    """按受试者划分训练/验证集，并进行类别分层。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["_subject_key"]].append(r)

    by_class: dict[int, list[str]] = defaultdict(list)
    for sk, clips in groups.items():
        by_class[label_from_row(clips[0], classes)].append(sk)

    rng = random.Random(seed)
    train_keys: list[str] = []
    val_keys: list[str] = []
    for lbl, keys in by_class.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n_val = max(1, int(round(len(keys) * val_split)))
        val_keys.extend(keys[:n_val])
        train_keys.extend(keys[n_val:])

    train_rows = [r for sk in train_keys for r in groups[sk]]
    val_rows = [r for sk in val_keys for r in groups[sk]]
    return train_rows, val_rows


def count_labels(rows: list[dict], classes: str = "binary") -> dict[str, int]:
    names = label_names(classes)
    counts: dict[str, int] = {n: 0 for n in names}
    for r in rows:
        counts[names[label_from_row(r, classes)]] += 1
    return counts
