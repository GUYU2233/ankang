#!/usr/bin/env python3
"""ultralytics YOLO26-pose 微调入口（P4 微调脚本）。

运行解释器：/home/ljh/ankang/ai-engine/.venv/bin/python（ultralytics 8.4.131 + torch 2.13.0+cpu，无 GPU）。
项目根 = Path(__file__).resolve().parents[1]（即 /home/ljh/ankang），一切相对路径基于它，不依赖 cwd。

【CLI】
  --model     预训练权重，默认 ai-engine/models/yolo26s-pose.pt（相对项目根解析，
              已实测可加载：task=pose、names={0:person}、kpt_shape=[17,3]）
  --data      dataset yaml，默认 data/annotations/pose_ft/pose_ft.yaml
  --epochs    默认 30；--batch 默认 4；--device 默认 cpu；--imgsz 默认 640
  --project   默认 ai-engine/runs/pose_ft；--name 默认 pose_ft_YYYYMMDD_HHMMSS
  --patience  可选（EarlyStopping 耐心轮数，缺省不传）

【实现】
  - import ultralytics 前 os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/Ultralytics')：
    实测 /root/.config/Ultralytics 不可写会自动回退 /tmp，setdefault 固定行为并尽量消除警告；
    顺带 setdefault MPLCONFIGDIR 避免 matplotlib 在只读 /root/.config 下反复告警。
  - 预检：data yaml 存在；train/val 目录图片数 > 0（否则 clear error + exit != 0）；
    统计 labels 覆盖率（有 txt 的图片比例）打印——仅警告不阻断，未标注图作背景参与训练。
  - 核心调用：YOLO(model).train(data=data, epochs=epochs, batch=batch, device=device,
    imgsz=imgsz, freeze=10, lr0=1e-3, cos_lr=True, rect=True, workers=2,
    project=project, name=name)。
  - 收尾：打印 best.pt/last.pt 绝对路径、训练耗时、最终 box mAP50/mAP50-95 与
    pose mAP50/mAP50-95（取返回 metrics），并提示 30 epoch CPU 全量训练耗时较长
    （预计小时级），Pilot 验收只跑 smoke。
"""

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# —— 必须先于 ultralytics import ——
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import yaml  # noqa: E402   (ultralytics 依赖 pyyaml，venv 中必有)

from ultralytics import YOLO  # noqa: E402


def resolve(p: str) -> Path:
    """相对路径一律基于项目根解析，不依赖 cwd。"""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def count_images(img_dir: Path):
    return len([p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])


def preflight(data_path: Path, model_path: Path):
    """预检；任一硬性失败 → 打印清晰错误并 exit(1)。返回 (data_dir, train_dir, val_dir)。"""
    if not model_path.is_file():
        print(f"[错误] 模型权重不存在: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not data_path.is_file():
        print(f"[错误] dataset yaml 不存在: {data_path}\n"
              f"       请先确认 data/annotations/pose_ft/pose_ft.yaml 已生成（或 --data 指定正确路径）。",
              file=sys.stderr)
        sys.exit(1)

    with open(data_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # 数据集根 = yaml 所在目录；忽略 yaml 里可能残留的旧机器绝对路径（可移植性关键）
    data_dir = data_path.parent
    train_dir = data_dir / cfg.get("train", "images/train")
    val_dir = data_dir / cfg.get("val", "images/val")
    n_train = count_images(train_dir)
    n_val = count_images(val_dir)
    if n_train == 0 or n_val == 0:
        print(f"[错误] yaml train/val 图片数为 0（train={n_train}, val={n_val}）:\n"
              f"       train -> {train_dir}\n       val   -> {val_dir}", file=sys.stderr)
        sys.exit(1)

    print("=== 预检 ===")
    print(f"  data yaml : {data_path}")
    print(f"  train 图片: {n_train}  ({train_dir})")
    print(f"  val   图片: {n_val}    ({val_dir})")
    for split, img_dir, n in (("train", train_dir, n_train), ("val", val_dir, n_val)):
        label_dir = data_dir / "labels" / split
        labeled = 0
        if label_dir.is_dir():
            labeled = sum(1 for p in img_dir.glob("*")
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                          and (label_dir / f"{p.stem}.txt").is_file())
        cov = labeled / n if n else 0.0
        flag = "" if cov >= 1.0 else "  [警告] 存在未标注图（作背景参与训练，不阻断）"
        print(f"  {split} labels 覆盖率: {labeled}/{n} ({cov*100:.1f}%){flag}")
    return data_dir, train_dir, val_dir


def main():
    parser = argparse.ArgumentParser(
        description="ultralytics YOLO26-pose 微调入口（P4 微调脚本，无 GPU 全 CPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="ai-engine/models/yolo26s-pose.pt",
                        help="预训练权重（相对项目根解析）")
    parser.add_argument("--data", default="data/annotations/pose_ft/pose_ft.yaml",
                        help="dataset yaml（相对项目根解析）")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--device", default="cpu", help="训练设备（无 GPU，默认 cpu）")
    parser.add_argument("--imgsz", type=int, default=640, help="训练/推理输入尺寸")
    parser.add_argument("--project", default="ai-engine/runs/pose_ft",
                        help="训练输出项目目录（相对项目根解析）")
    parser.add_argument("--name", default=f"pose_ft_{datetime.now():%Y%m%d_%H%M%S}",
                        help="本次训练 run 名（默认 pose_ft_YYYYMMDD_HHMMSS，每次运行自动取新时间戳）")
    parser.add_argument("--patience", type=int, default=None,
                        help="EarlyStopping 耐心轮数（缺省不启用）")
    args = parser.parse_args()

    model_path = resolve(args.model)
    data_path = resolve(args.data)
    project_path = resolve(args.project)
    name = args.name

    data_dir, _train_dir, _val_dir = preflight(data_path, model_path)

    # 可移植性：把 dataset yaml 的 path 覆盖为按项目根解析出的绝对路径，
    # 写入临时 yaml 再传给 ultralytics（它要求 data 为路径字符串），
    # 消除跨机器的硬编码绝对路径依赖，且不改动仓库内文件。
    with open(data_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}
    data_cfg["path"] = str(data_dir)
    _tmp_f = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="pose_ft_resolved_", delete=False, encoding="utf-8")
    yaml.safe_dump(data_cfg, _tmp_f, allow_unicode=True, sort_keys=False)
    resolved_data = _tmp_f.name
    _tmp_f.close()

    train_kwargs = dict(
        data=resolved_data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        imgsz=args.imgsz,
        freeze=10,
        lr0=1e-3,
        cos_lr=True,
        rect=True,
        workers=2,
        project=str(project_path),
        name=name,
    )
    if args.patience is not None:
        train_kwargs["patience"] = args.patience

    print("\n=== 训练配置 ===")
    for k, v in train_kwargs.items():
        print(f"  {k:<10} = {v}")

    t0 = time.time()
    try:
        model = YOLO(str(model_path))
        res = model.train(**train_kwargs)
    except KeyboardInterrupt:
        print("\n[中断] 用户中断训练（Ctrl-C）。", file=sys.stderr)
        sys.exit(130)
    finally:
        Path(resolved_data).unlink(missing_ok=True)
    elapsed = time.time() - t0

    run_dir = project_path / name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    print("\n=== 训练完成 ===")
    print(f"  best.pt : {best.resolve() if best.exists() else best} "
          f"({'存在' if best.exists() else '缺失!'})")
    print(f"  last.pt : {last.resolve() if last.exists() else last} "
          f"({'存在' if last.exists() else '缺失!'})")
    print(f"  results.csv: {run_dir / 'results.csv'}")
    print(f"  训练耗时: {int(elapsed // 60)}m {int(elapsed % 60)}s")

    try:
        print(f"  最终 box  mAP50={res.box.map50:.4f}  mAP50-95={res.box.map:.4f}")
        print(f"  最终 pose mAP50={res.pose.map50:.4f}  mAP50-95={res.pose.map:.4f}")
    except AttributeError:
        print("  [注意] 未能从 metrics 读取 mAP（训练可能未完成验证）。")

    print("\n[提示] 30 epoch CPU 全量训练耗时较长（预计小时级）；Pilot 验收请用 "
          "--epochs 1 --batch 2 --name smoke 跑通即可，全量训练由操作者另行决定。")


if __name__ == "__main__":
    main()
