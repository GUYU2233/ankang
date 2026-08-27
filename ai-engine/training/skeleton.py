"""骨架提取：用 YOLOv8-Pose 从视频抽帧并缓存 17 关键点序列。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from training.config import DataConfig


def sample_frame_indices(src_fps: float, frame_count: int, target_fps: float) -> list[int]:
    if frame_count <= 0:
        return []
    src_fps = max(src_fps, 1e-3)
    step = max(1.0, src_fps / max(target_fps, 1e-3))
    indices: list[int] = []
    t = 0.0
    while int(t) < frame_count:
        idx = int(t)
        if idx not in indices:
            indices.append(idx)
        t += step
    return indices


class SkeletonExtractor:
    def __init__(self, model_path: str | Path, device: str = "auto") -> None:
        self.model_path = str(model_path)
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def predict_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """对一批 BGR 帧做姿态估计，返回每帧的主目标 17x3 关键点。"""
        model = self._load()
        results = model.predict(frames, verbose=False, conf=0.3, device=self.device)
        out: list[np.ndarray] = []
        for res in results:
            kpts = self._main_person(res)
            out.append(kpts)
        return out

    @staticmethod
    def _main_person(res) -> np.ndarray:
        kp = res.keypoints
        if kp is None or kp.data is None or kp.data.shape[0] == 0:
            return np.zeros((17, 3), dtype=np.float32)
        data = kp.data.cpu().numpy()  # (N, 17, 3)
        conf = data[:, :, 2]
        mean_conf = conf.mean(axis=1)
        idx = int(np.argmax(mean_conf))
        return data[idx].astype(np.float32)

    def extract(self, video_path: str | Path, target_fps: float, conf_thr: float, batch_size: int = 64) -> np.ndarray:
        """抽取整段视频骨架，返回 (T, 17, 3)。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or target_fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(src_fps, frame_count, target_fps)

        all_kpts: list[np.ndarray] = []
        pos = 0
        while pos < len(indices):
            chunk_idx = indices[pos:pos + batch_size]
            frames: list[np.ndarray] = []
            valid = [False] * len(chunk_idx)
            for i, fidx in enumerate(chunk_idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
                    valid[i] = True
            if frames:
                kpts_batch = self.predict_batch(frames)
                ki = 0
                for i, is_valid in enumerate(valid):
                    if is_valid:
                        k = kpts_batch[ki]
                        ki += 1
                    else:
                        k = np.zeros((17, 3), dtype=np.float32)
                    k[k[:, 2] < conf_thr] = 0.0
                    all_kpts.append(k)
            else:
                all_kpts.extend([np.zeros((17, 3), dtype=np.float32)] * len(chunk_idx))
            pos += batch_size
        cap.release()

        if not all_kpts:
            return np.zeros((0, 17, 3), dtype=np.float32)
        return np.stack(all_kpts, axis=0).astype(np.float32)


def cache_path_for(row: dict, cfg: DataConfig) -> Path:
    video = Path(row["video_path"])
    return cfg.skeleton_cache / (video.stem + ".npz")


def load_cached(cache_path: Path) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    data = np.load(cache_path)
    return data["keypoints"].astype(np.float32)


def extract_and_cache(
    row: dict,
    cfg: DataConfig,
    extractor: SkeletonExtractor,
    batch_size: int = 64,
) -> Path:
    video_path = cfg.data_root / row["video_path"]
    cache_path = cache_path_for(row, cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    kpts = extractor.extract(video_path, cfg.target_fps, cfg.keypoint_conf_thr, batch_size)
    np.savez_compressed(cache_path, keypoints=kpts, target_fps=cfg.target_fps)
    return cache_path


def precompute_skeletons(
    rows: list[dict],
    cfg: DataConfig,
    batch_size: int = 64,
    skip_existing: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Path]:
    """为全部视频抽取并缓存骨架，返回 {video_stem: cache_path}。"""
    extractor = SkeletonExtractor(cfg.pose_model, cfg.device)
    cache_map: dict[str, Path] = {}
    total = len(rows)
    for i, row in enumerate(rows):
        video = Path(row["video_path"])
        cache_path = cache_path_for(row, cfg)
        if skip_existing and cache_path.exists():
            cache_map[video.stem] = cache_path
        else:
            try:
                extract_and_cache(row, cfg, extractor, batch_size)
                cache_map[video.stem] = cache_path
            except Exception as exc:
                print(f"skeleton skipped {row['video_path']}: {exc}", flush=True)
        if progress:
            progress(i + 1, total)
    return cache_map
