"""评测脚本：加载训练好的 checkpoint，在按受试者划分的验证集上输出指标。

用法：
    python -m training.evaluate --checkpoint ai-engine/runs/tcn_binary_w32.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_AI_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI_ROOT))

from training.config import DataConfig, ModelConfig
from training.dataset import build_dataset
from training.manifest import annotate_subject_keys, load_manifest, load_source_mapping, split_by_subject
from training.metrics import compute_metrics, format_metrics
from training.models import build_model
from training.train import evaluate, resolve_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window-len", type=int, default=32)
    ap.add_argument("--window-stride", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    ap.add_argument("--manifest", default=str(_REPO_ROOT / "data" / "manifest.csv"))
    ap.add_argument("--cache-dir", default=str(_REPO_ROOT / "data" / "annotations" / "skeleton"))
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    model_cfg_dict = ckpt.get("model_cfg", {})
    classes = ckpt.get("classes", "binary")
    class_names = ckpt.get("class_names", ["non_fall", "fall"] if classes == "binary" else ["fall", "nearfall", "normal"])
    window_len = ckpt.get("window_len", args.window_len)

    model_cfg = ModelConfig(**{**model_cfg_dict, "num_classes": len(class_names)})
    device = resolve_device(args.device)
    model = build_model(model_cfg).to(device)
    model.load_state_dict(state_dict)
    print(f"loaded {args.checkpoint} model={model_cfg.name} classes={classes}")

    data_cfg = DataConfig(
        data_root=Path(args.data_root),
        manifest=Path(args.manifest),
        skeleton_cache=Path(args.cache_dir),
        window_len=window_len,
        window_stride=args.window_stride,
        device=device,
    )
    rows = load_manifest(data_cfg.manifest)
    annotate_subject_keys(rows, load_source_mapping(data_cfg.source_mapping))
    rows = [r for r in rows if r.get("action_label", "normal") in ("fall", "nearfall", "normal", "risk_behavior")
            and (data_cfg.data_root / r["video_path"]).exists()]
    _, val_rows = split_by_subject(rows, args.val_split, args.seed, classes)

    val_ds = build_dataset(val_rows, data_cfg, classes)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    _, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    metrics = compute_metrics(y_true, y_pred, len(class_names))
    print("\n== validation ==")
    print(format_metrics(metrics, class_names))


if __name__ == "__main__":
    main()
