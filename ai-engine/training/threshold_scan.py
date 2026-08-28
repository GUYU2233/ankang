"""阈值扫描与事件级评估。

按受试者划分的验证集上，将训练好的 TCN 作为流式分类器在完整视频骨架上滑窗，
输出窗口级指标、跌倒事件召回、误报频率、检测延迟，并扫描业务阈值。

用法：
    python -m training.threshold_scan --checkpoint ai-engine/runs/tcn_binary_w32.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_AI_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI_ROOT))

from training.config import DataConfig, ModelConfig
from training.dataset import normalize_sequence
from training.manifest import annotate_subject_keys, load_manifest, load_source_mapping, split_by_subject
from training.models import build_model

FALL_START = "fall_start_ts"
FALL_END = "fall_end_ts"


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    model_cfg_dict = ckpt.get("model_cfg", {})
    class_names = ckpt.get("class_names", ["non_fall", "fall"])
    window_len = ckpt.get("window_len", 32)
    model_cfg = ModelConfig(**{**model_cfg_dict, "num_classes": len(class_names)})
    model = build_model(model_cfg)
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_cfg, class_names, window_len


def prob_timeline(model, kpts_norm: np.ndarray, window_len: int, device: str) -> np.ndarray:
    """滑窗输出每一帧对应的跌倒概率。kpts_norm: (T,17,3) 已归一化。"""
    t = kpts_norm.shape[0]
    out = np.zeros(t, dtype=np.float32)
    if t == 0:
        return out
    if t < window_len:
        pad = np.zeros((window_len - t, 17, 3), dtype=np.float32)
        seq = np.concatenate([kpts_norm, pad], axis=0)
        x = torch.from_numpy(np.transpose(seq, (2, 0, 1))[None]).float().to(device)
        with torch.no_grad():
            logits = model(x)[0].cpu().numpy()
        p = float(np.exp(logits[1] - np.max(logits)) / np.sum(np.exp(logits - np.max(logits))))
        out[:] = p
        return out
    starts = np.arange(0, t - window_len + 1)
    probs = np.zeros(t, dtype=np.float32)
    total = len(starts)
    # 批量切窗
    batch = []
    idxs = []
    # 逐帧概率：把窗口概率贴到窗口中心帧
    half = window_len // 2
    for si in range(0, total, 256):
        chunk = starts[si:si + 256]
        xs = np.stack([np.transpose(kpts_norm[s:s + window_len], (2, 0, 1)) for s in chunk], axis=0)
        xt = torch.from_numpy(xs).float().to(device)
        with torch.no_grad():
            lg = model(xt).cpu().numpy()
        p = 1.0 / (1.0 + np.exp(lg[:, 0] - lg[:, 1]))
        for j, s in enumerate(chunk):
            center = s + half
            probs[center] = max(probs[center], p[j])
    # 用最近窗口概率填充首尾未覆盖帧
    first_center = starts[0] + half
    last_center = starts[-1] + half
    probs[np.arange(first_center)] = probs[first_center]
    probs[np.arange(last_center + 1, t)] = probs[last_center]
    return probs


def count_false_alarms(probs: np.ndarray, thr: float, fps: float, merge_gap_frames: int) -> int:
    """把连续超阈值片段合并为一次误报，间隔超过 merge_gap 视为两次。"""
    flag = probs >= thr
    events = 0
    last_end = -10 ** 9
    i = 0
    n = len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j < n and flag[j]:
                j += 1
            if i - last_end > merge_gap_frames:
                events += 1
            last_end = j
            i = j
        else:
            i += 1
    return events


def evaluate_events(rows, model, window_len, device, thr, fps, merge_gap_frames, cache_root):
    fall_records = []
    nonfall_records = []
    total_nonfall_duration = 0.0
    scenes = {}

    for r in rows:
        action = r.get("action_label", "normal")
        kpts = load_keypoints(r, cache_root)
        if kpts is None or kpts.shape[0] == 0:
            continue
        kpts_norm = normalize_sequence(kpts, 0.3)
        probs = prob_timeline(model, kpts_norm, window_len, device)
        dur = kpts.shape[0] / fps
        scene = r.get("scene", "unknown")
        rec = {"video": r["video_path"], "scene": scene, "prob": float(probs.max()), "duration_s": round(dur, 2)}

        if action == "fall":
            fs = float(r.get(FALL_START) or 0.0)
            fe = float(r.get(FALL_END) or 0.0)
            if fe <= fs:
                # 无区间标注时退化为整段
                fs, fe = 0.0, dur
            a, b = int(round(fs * fps)), int(round(fe * fps))
            a = max(0, min(a, len(probs) - 1))
            b = max(a, min(b, len(probs) - 1))
            seg = probs[a:b + 1]
            detected = bool((seg >= thr).any())
            peak_rel_s = float(np.argmax(seg) / fps) if detected else None
            delay_from_start = peak_rel_s
            rec.update({"detected": detected, "fall_start_s": fs, "delay_s": round(delay_from_start, 2) if detected else None})
            fall_records.append(rec)
        else:
            alarms = count_false_alarms(probs, thr, fps, merge_gap_frames)
            rec["false_alarms"] = alarms
            total_nonfall_duration += dur
            nonfall_records.append(rec)

        scenes.setdefault(scene, {"fall": 0, "detected": 0, "nonfall": 0, "false_alarms": 0})
        if action == "fall":
            scenes[scene]["fall"] += 1
            scenes[scene]["detected"] += 1 if rec.get("detected") else 0
        else:
            scenes[scene]["nonfall"] += 1
            scenes[scene]["false_alarms"] += rec["false_alarms"]

    event_recall = sum(1 for r in fall_records if r["detected"]) / max(len(fall_records), 1)
    total_alarms = sum(r["false_alarms"] for r in nonfall_records)
    false_alarms_per_hour = total_alarms / (total_nonfall_duration / 3600.0) if total_nonfall_duration > 0 else 0.0
    delays = [r["delay_s"] for r in fall_records if r["delay_s"] is not None]
    median_delay = float(np.median(delays)) if delays else None
    p95_delay = float(np.percentile(delays, 95)) if delays else None

    scene_metrics = {
        k: {
            "fall": v["fall"], "detected": v["detected"],
            "event_recall": round(v["detected"] / v["fall"], 3) if v["fall"] else None,
            "nonfall": v["nonfall"], "false_alarms": v["false_alarms"],
        }
        for k, v in scenes.items()
    }
    return {
        "threshold": thr,
        "n_fall_videos": len(fall_records),
        "n_nonfall_videos": len(nonfall_records),
        "event_recall": round(event_recall, 4),
        "false_alarms_per_hour": round(false_alarms_per_hour, 4),
        "median_delay_s": median_delay,
        "p95_delay_s": p95_delay,
        "scene": scene_metrics,
        "fall_records": fall_records,
        "nonfall_records": nonfall_records,
    }


def load_keypoints(row, cache_root):
    from training.skeleton import load_cached
    video = Path(row["video_path"])
    return load_cached(cache_root / (video.stem + ".npz"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window-len", type=int, default=32)
    ap.add_argument("--target-fps", type=float, default=15.0)
    ap.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    ap.add_argument("--manifest", default=str(_REPO_ROOT / "data" / "manifest.csv"))
    ap.add_argument("--cache-dir", default=str(_REPO_ROOT / "data" / "annotations" / "skeleton"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--min-thr", type=float, default=0.20)
    ap.add_argument("--max-thr", type=float, default=0.95)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", default=str(_REPO_ROOT / "ai-engine" / "runs"))
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    model, model_cfg, class_names, window_len = load_checkpoint(args.checkpoint)
    model = model.to(device)

    data_cfg = DataConfig(data_root=Path(args.data_root), manifest=Path(args.manifest), skeleton_cache=Path(args.cache_dir), target_fps=args.target_fps)
    rows = load_manifest(data_cfg.manifest)
    annotate_subject_keys(rows, load_source_mapping(data_cfg.source_mapping))
    rows = [
        r for r in rows
        if r.get("action_label", "normal") in ("fall", "nearfall", "normal", "risk_behavior")
        and (data_cfg.data_root / r["video_path"]).exists()
    ]
    _, val_rows = split_by_subject(rows, args.val_split, args.seed, "binary")

    cache_root = data_cfg.skeleton_cache
    fps = args.target_fps
    merge_gap_frames = int(fps)

    thresholds = [round(t, 2) for t in np.arange(args.min_thr, args.max_thr + 1e-9, args.step)]
    sweep = []
    best = None
    for thr in thresholds:
        res = evaluate_events(val_rows, model, window_len, device, thr, fps, merge_gap_frames, cache_root)
        recall = res["event_recall"]
        far = res["false_alarms_per_hour"]
        # 综合得分：召回为主，误报惩罚
        score = recall - 0.15 * (far / 60.0)
        row = {
            "threshold": thr,
            "event_recall": recall,
            "false_alarms_per_hour": far,
            "median_delay_s": res["median_delay_s"],
            "p95_delay_s": res["p95_delay_s"],
            "score": round(score, 4),
        }
        sweep.append(row)
        if best is None or score > best["score"]:
            best = row
        print(f"thr={thr:.2f}  recall={recall:.3f}  FAR/h={far:.2f}  median_delay={res['median_delay_s']}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "threshold_sweep.json").write_text(json.dumps(sweep, ensure_ascii=False, indent=2), encoding="utf-8")

    recommended = best["threshold"]
    final = evaluate_events(val_rows, model, window_len, device, recommended, fps, merge_gap_frames, cache_root)
    final.pop("fall_records", None)
    final.pop("nonfall_records", None)
    (out_dir / "event_metrics.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== recommended threshold ==")
    print(json.dumps({"threshold": recommended, "sweep": best}, ensure_ascii=False, indent=2))
    print("\n== event metrics ==")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
