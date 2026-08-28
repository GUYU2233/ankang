"""训练数据与标签不变量测试。

用法：python ai-engine/tests/test_training_invariants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.manifest import (
    annotate_subject_keys,
    label_from_row,
    load_manifest,
    load_source_mapping,
    split_by_subject,
)
from training.train import is_decodable_video

DATA = Path(__file__).resolve().parents[2]


def test_subject_split_no_leak() -> None:
    rows = load_manifest(DATA / "data" / "manifest.csv")
    mapping = load_source_mapping(DATA / "data" / "meta" / "source_mapping.csv")
    annotate_subject_keys(rows, mapping)
    rows = [r for r in rows if (DATA / "data" / r["video_path"]).exists()]
    train, val = split_by_subject(rows, 0.2, 42, "binary")
    tr_keys = {r["_subject_key"] for r in train}
    va_keys = {r["_subject_key"] for r in val}
    assert not (tr_keys & va_keys), "训练/验证受试者存在泄漏"
    print(f"  split no-leak OK: train={len(tr_keys)} val={len(va_keys)}")


def test_risk_behavior_maps_nonfall() -> None:
    assert label_from_row({"action_label": "risk_behavior"}, "binary") == 0
    assert label_from_row({"action_label": "fall"}, "binary") == 1
    assert label_from_row({"action_label": "nearfall"}, "binary") == 0
    assert label_from_row({"action_label": "risk_behavior"}, "ternary") == 2
    print("  label mapping OK")


def test_lfs_pointer_rejected() -> None:
    tmp = Path("__lfs_pointer_probe.bin")
    tmp.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 100\n", encoding="ascii")
    try:
        assert is_decodable_video(tmp) is False, "LFS 指针应判定不可解码"
    finally:
        tmp.unlink(missing_ok=True)
    print("  lfs pointer rejection OK")


def test_real_video_decodable() -> None:
    rows = load_manifest(DATA / "data" / "manifest.csv")
    good = next(
        r for r in rows
        if (DATA / "data" / r["video_path"]).exists() and float(r.get("fps") or 0) > 0
    )
    assert is_decodable_video(DATA / "data" / good["video_path"]) is True
    print("  real video decodable OK:", good["video_path"])


def main() -> None:
    test_subject_split_no_leak()
    test_risk_behavior_maps_nonfall()
    test_lfs_pointer_rejected()
    test_real_video_decodable()
    print("TRAINING INVARIANTS OK")


if __name__ == "__main__":
    main()
