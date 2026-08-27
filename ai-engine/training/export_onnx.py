"""从 checkpoint 重新导出 ONNX。

用法：
    python -m training.export_onnx --checkpoint ai-engine/runs/tcn_binary_w32.pt --out model.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_AI_ROOT = Path(__file__).resolve().parents[1]
if str(_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI_ROOT))

from training.config import ModelConfig
from training.models import build_model
from training.train import export_onnx, resolve_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    model_cfg_dict = ckpt.get("model_cfg", {})
    class_names = ckpt.get("class_names", ["non_fall", "fall"])
    window_len = ckpt.get("window_len", 32)

    model_cfg = ModelConfig(**{**model_cfg_dict, "num_classes": len(class_names)})
    device = resolve_device(args.device)
    model = build_model(model_cfg).to(device)
    model.load_state_dict(state_dict)

    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".onnx")
    export_onnx(model, model_cfg.in_channels, model_cfg.num_joints, window_len, out, device)
    print(f"exported: {out}")


if __name__ == "__main__":
    main()
