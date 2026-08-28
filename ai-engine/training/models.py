"""时序分类模型：TCN 与轻量 ST-GCN（纯 PyTorch，不依赖 torch_geometric）。

输入统一为 (N, C, T, V)：C=3(x,y,conf)、T=窗口帧数、V=17(COCO 关键点)。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from training.config import ModelConfig

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def build_adjacency(num_joints: int = 17) -> torch.Tensor:
    a = np.eye(num_joints, dtype=np.float32)
    for i, j in COCO_EDGES:
        if i < num_joints and j < num_joints:
            a[i, j] = 1.0
            a[j, i] = 1.0
    d = np.sum(a, axis=1) ** -0.5
    d = np.diag(d)
    a_norm = d @ a @ d
    return torch.tensor(a_norm, dtype=torch.float32)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_joints: int,
        num_classes: int,
        hidden: int = 128,
        layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_joints = num_joints
        in_channels = in_dim * num_joints
        blocks = []
        prev = in_channels
        for i in range(layers):
            blocks.append(TemporalBlock(prev, hidden, kernel_size, 2 ** i, dropout))
            prev = hidden
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(prev, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = x.permute(0, 1, 3, 2).reshape(n, c * v, t)
        x = self.blocks(x)
        return self.head(x)


class SpatialGraphConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, a: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("a", a)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V) -> message passing over V
        x = x.permute(0, 2, 3, 1)  # (N,T,V,C)
        x = torch.einsum("ij,ntic->ntjc", self.a, x)
        x = x.permute(0, 3, 1, 2)  # (N,C,T,V)
        return self.conv(x)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, a: torch.Tensor, dropout: float = 0.3) -> None:
        super().__init__()
        self.spatial = SpatialGraphConv(in_ch, out_ch, a)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.temporal = nn.Conv2d(out_ch, out_ch, kernel_size=(9, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.spatial(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.temporal(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return self.relu(out + res)


class STGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_joints: int,
        num_classes: int,
        hidden: int = 128,
        blocks: int = 6,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        a = build_adjacency(num_joints)
        layers = []
        prev = in_dim
        for _ in range(blocks):
            layers.append(STGCNBlock(prev, hidden, a, dropout))
            prev = hidden
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(prev, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.layers(x))


class PrefallRiskHead(nn.Module):
    """可训练的前置风险头：4 类 logits (fall / nearfall / risk_behavior / normal)。

    当前在线推理仍使用启发式 HeuristicPoseRiskHead；本模块作为训练接口。
    导出 ONNX 后可替换 ai-engine/app/pipelines/prefall_head.py。
    输入 (N, C, T, V)，输出 (N, 4)。
    """

    def __init__(
        self,
        in_dim: int = 3,
        num_joints: int = 17,
        hidden: int = 128,
        layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = TCN(
            in_dim=in_dim,
            num_joints=num_joints,
            num_classes=num_classes,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.name == "prefall":
        return PrefallRiskHead(
            in_dim=cfg.in_channels,
            num_joints=cfg.num_joints,
            hidden=cfg.hidden,
            layers=max(4, cfg.tcn_layers - 2),
            dropout=cfg.dropout,
            num_classes=cfg.num_classes,
        )
    if cfg.name == "stgcn":
        return STGCN(
            in_dim=cfg.in_channels,
            num_joints=cfg.num_joints,
            num_classes=cfg.num_classes,
            hidden=cfg.hidden,
            blocks=cfg.stgcn_blocks,
            dropout=cfg.dropout,
        )
    return TCN(
        in_dim=cfg.in_channels,
        num_joints=cfg.num_joints,
        num_classes=cfg.num_classes,
        hidden=cfg.hidden,
        layers=cfg.tcn_layers,
        kernel_size=cfg.tcn_kernel,
        dropout=cfg.dropout,
    )
