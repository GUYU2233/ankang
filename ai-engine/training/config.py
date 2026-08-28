"""训练配置：路径、数据窗口、模型与训练超参。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_root: Path = Path("data")
    manifest: Path = Path("data/manifest.csv")
    source_mapping: Path = Path("data/meta/source_mapping.csv")
    skeleton_cache: Path = Path("data/annotations/skeleton")
    pose_model: Path = Path("ai-engine/models/yolov8n-pose.pt")
    target_fps: float = 15.0
    window_len: int = 32
    window_stride: int = 16
    keypoint_conf_thr: float = 0.3
    max_persons: int = 1
    device: str = "auto"  # auto / cpu / cuda


@dataclass
class ModelConfig:
    name: str = "tcn"  # tcn / stgcn
    num_classes: int = 2
    in_channels: int = 3  # x, y, conf
    num_joints: int = 17
    hidden: int = 128
    dropout: float = 0.3
    tcn_layers: int = 6
    tcn_kernel: int = 3
    stgcn_blocks: int = 6


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2
    seed: int = 42
    num_workers: int = 0
    output_dir: Path = Path("ai-engine/runs")
    class_weight: bool = True
    classes: str = "binary"  # binary / ternary / multiclass
    max_clips: int = 0  # 0 = all clips (方便冒烟测试)
