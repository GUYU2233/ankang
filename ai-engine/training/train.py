"""基线训练脚本：加载骨架 -> 时序窗口 -> 训练 TCN/ST-GCN -> 评测 -> 导出 ONNX。

用法（在 ai-engine/ 目录下）：
    python -m training.train --model tcn --epochs 30 --max-clips 0
    python -m training.train --model stgcn --classes ternary
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_AI_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI_ROOT))

from training.config import DataConfig, ModelConfig, TrainConfig
from training.dataset import build_dataset
from training.manifest import (
    annotate_subject_keys,
    count_labels,
    label_from_row,
    label_names,
    load_manifest,
    load_source_mapping,
    split_by_subject,
)
from training.metrics import compute_metrics, format_metrics
from training.models import build_model
from training.skeleton import precompute_skeletons


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_decodable_video(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.stat().st_size < 1024:
        try:
            if path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
                return False
        except OSError:
            return False
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    if ok:
        ok, _ = cap.read()
    cap.release()
    return bool(ok)


def compute_class_weights(train_rows: list[dict], num_classes: int, classes: str) -> torch.Tensor:
    counts = [0] * num_classes
    for r in train_rows:
        counts[label_from_row(r, classes)] += 1
    counts = [max(c, 1) for c in counts]
    total = sum(counts)
    weights = [total / (num_classes * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.shape[0]
        n += x.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    n = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.shape[0]
        n += x.shape[0]
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / max(n, 1), y_true, y_pred


def export_onnx(model, in_dim: int, num_joints: int, window_len: int, path: Path, device: str) -> None:
    model.eval()
    dummy = torch.zeros(1, in_dim, window_len, num_joints, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["skeleton"],
        output_names=["logits"],
        dynamic_axes={"skeleton": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def save_confusion_png(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tcn", "stgcn"], default="tcn")
    ap.add_argument("--classes", choices=["binary", "ternary"], default="binary")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window-len", type=int, default=32)
    ap.add_argument("--window-stride", type=int, default=16)
    ap.add_argument("--target-fps", type=float, default=15.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--skip-skeleton", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    ap.add_argument("--manifest", default=str(_REPO_ROOT / "data" / "manifest.csv"))
    ap.add_argument("--pose-model", default=str(_REPO_ROOT / "ai-engine" / "models" / "yolov8n-pose.pt"))
    ap.add_argument("--cache-dir", default=str(_REPO_ROOT / "data" / "annotations" / "skeleton"))
    ap.add_argument("--output-dir", default=str(_REPO_ROOT / "ai-engine" / "runs"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}")

    data_cfg = DataConfig(
        data_root=Path(args.data_root),
        manifest=Path(args.manifest),
        skeleton_cache=Path(args.cache_dir),
        pose_model=Path(args.pose_model),
        target_fps=args.target_fps,
        window_len=args.window_len,
        window_stride=args.window_stride,
        device=device,
    )
    num_classes = 3 if args.classes == "ternary" else 2
    model_cfg = ModelConfig(name=args.model, num_classes=num_classes, in_channels=3, num_joints=17)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        classes=args.classes,
        max_clips=args.max_clips,
    )

    rows = load_manifest(data_cfg.manifest)
    mapping = load_source_mapping(data_cfg.source_mapping)
    annotate_subject_keys(rows, mapping)

    # 只保留视频存在且标签可映射的记录；risk_behavior 作为非跌倒/正常风险行为。
    valid = []
    for r in rows:
        if r.get("action_label", "normal") not in ("fall", "nearfall", "normal", "risk_behavior"):
            continue
        video_path = data_cfg.data_root / r["video_path"]
        if not is_decodable_video(video_path):
            continue
        valid.append(r)
    skipped = len(rows) - len(valid)
    rows = valid
    print(f"decodable clips={len(rows)} skipped={skipped}")

    if train_cfg.max_clips and 0 < train_cfg.max_clips < len(rows):
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: train_cfg.max_clips]

    train_rows, val_rows = split_by_subject(rows, train_cfg.val_split, train_cfg.seed, args.classes)
    names = label_names(args.classes)
    print(f"clips train={len(train_rows)} val={len(val_rows)}")
    print(f"train counts={count_labels(train_rows, args.classes)}")
    print(f"val counts={count_labels(val_rows, args.classes)}")

    print("precomputing skeletons ...")
    t0 = time.time()
    all_rows = train_rows + val_rows
    def report_progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"skeleton {done}/{total}", flush=True)

    precompute_skeletons(all_rows, data_cfg, skip_existing=args.skip_skeleton, progress=report_progress)
    print(f"skeleton done in {time.time() - t0:.1f}s")

    train_ds = build_dataset(train_rows, data_cfg, args.classes)
    val_ds = build_dataset(val_rows, data_cfg, args.classes)
    print(f"windows train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)

    model = build_model(model_cfg).to(device)
    class_weights = compute_class_weights(train_rows, num_classes, args.classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights if train_cfg.class_weight else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    out_dir = train_cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.model}_{args.classes}_w{args.window_len}"
    audit = {
        "manifest_rows": len(rows),
        "skipped_rows": skipped,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_subjects": sorted({r["_subject_key"] for r in train_rows}),
        "val_subjects": sorted({r["_subject_key"] for r in val_rows}),
        "train_label_counts": count_labels(train_rows, args.classes),
        "val_label_counts": count_labels(val_rows, args.classes),
    }
    (out_dir / f"{run_name}.audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    best_f1 = -1.0
    best_state = None
    best_metrics = None

    for epoch in range(1, train_cfg.epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        metrics = compute_metrics(y_true, y_pred, num_classes)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
        print(
            f"epoch {epoch:3d}/{train_cfg.epochs}  train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  "
            f"acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}"
        )

    print("\n== best validation ==")
    print(format_metrics(best_metrics, names))
    model.load_state_dict(best_state)

    ckpt_path = out_dir / f"{run_name}.pt"
    torch.save(
        {"model_cfg": model_cfg.__dict__, "state_dict": best_state, "window_len": args.window_len, "classes": args.classes, "class_names": names},
        ckpt_path,
    )
    print(f"saved checkpoint: {ckpt_path}")

    metrics_path = out_dir / f"{run_name}.metrics.json"
    metrics_path.write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_confusion_png(np.array(best_metrics["confusion_matrix"]), names, out_dir / f"{run_name}.confusion.png")

    if not args.no_export:
        onnx_path = out_dir / f"{run_name}.onnx"
        try:
            export_onnx(model, model_cfg.in_channels, model_cfg.num_joints, args.window_len, onnx_path, device)
            print(f"exported onnx: {onnx_path}")
        except Exception as exc:
            print(f"onnx export skipped (pip install onnx 后可用 export_onnx.py 重新导出): {exc}")


if __name__ == "__main__":
    main()
