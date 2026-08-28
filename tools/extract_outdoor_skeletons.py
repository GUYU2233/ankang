import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "ai-engine")
from training.config import DataConfig
from training.manifest import load_manifest
from training.skeleton import SkeletonExtractor, cache_path_for

cfg = DataConfig(
    data_root=Path("data"),
    manifest=Path("data/manifest.csv"),
    skeleton_cache=Path("data/annotations/skeleton"),
    device="cuda",
)
rows = [r for r in load_manifest(cfg.manifest) if r["scene"] == "outdoor"]
ex = SkeletonExtractor(cfg.pose_model, cfg.device)
for r in rows:
    c = cache_path_for(r, cfg)
    c.parent.mkdir(parents=True, exist_ok=True)
    kpts = ex.extract(cfg.data_root / r["video_path"], cfg.target_fps, cfg.keypoint_conf_thr, 64)
    np.savez_compressed(c, keypoints=kpts, target_fps=cfg.target_fps)
    print(r["video_path"], kpts.shape)
