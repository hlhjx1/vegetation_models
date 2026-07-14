#!/usr/bin/env python3
"""Formal four-dataset frozen DINOv3 vegetation segmentation training.

This reuses the DINOv2 four-dataset training/evaluation loop, but loads a
local DINOv3 ViT-L/16 SAT-493M backbone and writes outputs under 10_DINOv3.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

import run_dinov2_four_dataset_train as base


SAT493M_MEAN = torch.tensor([0.430, 0.411, 0.296]).view(3, 1, 1)
SAT493M_STD = torch.tensor([0.213, 0.156, 0.143]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen DINOv3 heads on four vegetation datasets.")
    parser.add_argument("--project-root", type=Path, default=Path("/content/vegetation_models_v2"))
    parser.add_argument("--binary-root", type=Path, default=Path("/content/binary"))
    parser.add_argument("--datasets", nargs="+", default=list(base.DATASET_DEFAULTS))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--torch-cache", type=Path, default=None)
    parser.add_argument("--dinov3-model", default="dinov3_vitl16")
    parser.add_argument("--dinov3-repo", type=Path, default=Path("/content/vegetation_models_v2/10_DINOv3/code/dinov3-main"))
    parser.add_argument(
        "--dinov3-weights",
        type=Path,
        default=Path("/content/vegetation_models_v2/10_DINOv3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
    )
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-predictions", type=int, default=12)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    parser.add_argument("--cache-in-ram", action="store_true", help="Cache resized tensors for each dataset in system RAM.")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last memory format on CUDA.")
    parser.add_argument("--allow-tf32", action="store_true", help="Allow TF32 matmul/convolution on CUDA.")
    parser.add_argument("--no-pretrained", action="store_true", help="Debug only: instantiate hub model without weights.")
    parser.add_argument(
        "--allow-zijinshan-random-split",
        action="store_true",
        help="Debug only: allow Zijinshan random train/val if train.txt/test.txt are missing.",
    )
    args = parser.parse_args()

    # Compatibility fields consumed by the shared DINOv2 training loop.
    args.dinov2_model = args.dinov3_model
    args.dinov2_repo = str(args.dinov3_repo)
    args.check_dinov3_entry = False
    return args


def load_backbone(args: argparse.Namespace, device: torch.device):
    if args.torch_cache:
        os.environ["TORCH_HOME"] = str(args.torch_cache)
    repo_path = Path(args.dinov3_repo).expanduser()
    if not repo_path.exists():
        raise FileNotFoundError(f"Missing DINOv3 repo: {repo_path}")
    if args.no_pretrained:
        backbone = torch.hub.load(str(repo_path), args.dinov3_model, pretrained=False, source="local")
    else:
        weights_path = Path(args.dinov3_weights).expanduser()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing DINOv3 weights: {weights_path}")
        backbone = torch.hub.load(str(repo_path), args.dinov3_model, source="local", weights=str(weights_path))
    backbone = backbone.to(device).eval()
    dummy = torch.zeros(1, 3, args.input_size, args.input_size, device=device)
    with torch.no_grad():
        features = backbone.forward_features(dummy)
        tokens = features["x_norm_patchtokens"] if isinstance(features, dict) else features
    _, patches, channels = tokens.shape
    side = int(round(patches**0.5))
    return backbone, int(channels), (side, patches // max(side, 1))


def main() -> int:
    base.IMAGENET_MEAN = SAT493M_MEAN
    base.IMAGENET_STD = SAT493M_STD
    base.parse_args = parse_args
    base.load_backbone = load_backbone

    original_main = base.main
    args = parse_args()
    if args.output_root is None:
        args.output_root = args.project_root / "10_DINOv3" / "four_dataset_runs"
    if args.results_csv is None:
        args.results_csv = args.project_root / "results" / "dinov3_four_dataset_results.csv"

    def fixed_parse_args() -> argparse.Namespace:
        return args

    base.parse_args = fixed_parse_args
    return original_main()


if __name__ == "__main__":
    raise SystemExit(main())
