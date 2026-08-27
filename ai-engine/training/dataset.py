"""PyTorch 数据集：把缓存骨架切成时序窗口，输出 (C, T, V) 张量。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from training.config import DataConfig
from training.manifest import label_from_row
from training.skeleton import load_cached


def normalize_frame(kpts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """单帧归一化：以髋中心为原点、躯干长度为尺度，做视角/尺度不变变换。"""
    conf = kpts[:, 2]
    valid = conf > 0.0
    xy = kpts[:, :2].copy()

    if valid[5] and valid[6] and valid[11] and valid[12]:
        mid_shoulder = (xy[5] + xy[6]) / 2.0
        mid_hip = (xy[11] + xy[12]) / 2.0
        center = mid_hip
        scale = float(np.linalg.norm(mid_shoulder - mid_hip))
    else:
        pts = xy[valid]
        if pts.shape[0] == 0:
            return kpts.astype(np.float32)
        center = pts.mean(axis=0)
        scale = float(np.max(np.linalg.norm(pts - center, axis=1)))

    if scale < eps:
        scale = 1.0

    out = np.zeros_like(kpts, dtype=np.float32)
    out[:, :2] = (xy - center) / scale
    out[:, 2] = conf
    out[~valid] = 0.0
    return out


def normalize_sequence(kpts: np.ndarray, conf_thr: float) -> np.ndarray:
    kpts = kpts.astype(np.float32)
    kpts[kpts[:, :, 2] < conf_thr] = 0.0
    return np.stack([normalize_frame(f) for f in kpts], axis=0)


def load_clip(row: dict, cfg: DataConfig) -> dict | None:
    video = Path(row["video_path"])
    cache_path = cfg.skeleton_cache / (video.stem + ".npz")
    kpts = load_cached(cache_path)
    if kpts is None or kpts.shape[0] == 0:
        return None
    return {
        "keypoints": kpts,
        "label": label_from_row(row, "binary"),
        "video": str(row["video_path"]),
        "subject": row.get("_subject_key", "unknown"),
        "action_label": row.get("action_label", "normal"),
    }


class SkeletonSequenceDataset(Dataset):
    def __init__(
        self,
        clips: list[dict],
        window_len: int,
        window_stride: int,
        conf_thr: float,
        classes: str = "binary",
    ) -> None:
        self.window_len = window_len
        self.classes = classes
        self.samples: list[tuple[np.ndarray, int]] = []
        for clip in clips:
            kpts = normalize_sequence(clip["keypoints"], conf_thr)
            label = clip["label"]
            t = kpts.shape[0]
            if t == 0:
                continue
            if t < window_len:
                pad = np.zeros((window_len - t, kpts.shape[1], kpts.shape[2]), dtype=np.float32)
                self.samples.append((np.concatenate([kpts, pad], axis=0), label))
            else:
                start = 0
                while start + window_len <= t:
                    self.samples.append((kpts[start:start + window_len], label))
                    start += window_stride
                if start < t and (t - window_len) > (start - window_stride):
                    self.samples.append((kpts[t - window_len:t], label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        kpts, label = self.samples[idx]
        # (T, 17, 3) -> (3, T, 17)
        x = torch.from_numpy(kpts).permute(2, 0, 1).float()
        return x, label


def build_dataset(
    rows: list[dict],
    cfg: DataConfig,
    classes: str = "binary",
) -> SkeletonSequenceDataset:
    clips: list[dict] = []
    for row in rows:
        clip = load_clip(row, cfg)
        if clip is not None:
            if classes == "ternary":
                clip["label"] = label_from_row(row, "ternary")
            clips.append(clip)
    return SkeletonSequenceDataset(clips, cfg.window_len, cfg.window_stride, cfg.keypoint_conf_thr, classes)
