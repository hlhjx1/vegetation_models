#!/usr/bin/env python3
"""Smoke-train a frozen DINOv2 ViT-L backbone with a lightweight segmentation head.

The script is intentionally small and conservative: it validates VOC-style data
loading, loss, mIoU calculation, and mask export on the 2024 vegetation dataset.
It is not a formal long-training run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
VEGETATION_RGB = np.array([144, 32, 192], dtype=np.uint8)


@dataclass(frozen=True)
class Sample:
    name: str
    image: Path
    mask: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen DINOv2 segmentation smoke training.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--torch-cache", type=Path, default=None)
    parser.add_argument("--dinov2-model", default="dinov2_vitl14_reg")
    parser.add_argument(
        "--dinov2-repo",
        default="facebookresearch/dinov2",
        help="Torch hub repo id, or a local dinov2 repo directory containing hubconf.py.",
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=12)
    parser.add_argument("--max-val-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="Debug only: instantiate hub model without weights.")
    parser.add_argument("--save-predictions", type=int, default=6)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_samples(dataset_root: Path) -> list[Sample]:
    split_path = dataset_root / "ImageSets" / "Segmentation" / "default.txt"
    image_dir = dataset_root / "JPEGImages"
    mask_dir = dataset_root / "SegmentationClass"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    samples: list[Sample] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        image_path = image_dir / f"{name}.png"
        mask_path = mask_dir / f"{name}.png"
        if image_path.is_file() and mask_path.is_file():
            samples.append(Sample(name, image_path, mask_path))
    if not samples:
        raise RuntimeError(f"No valid image/mask pairs found in {dataset_root}")
    return samples


class VegetationDataset(Dataset):
    def __init__(self, samples: list[Sample], input_size: int) -> None:
        self.samples = samples
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = Image.open(sample.image).convert("RGB").resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        mask_rgb = Image.open(sample.mask).convert("RGB").resize((self.input_size, self.input_size), Image.Resampling.NEAREST)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask_rgb, dtype=np.uint8)
        mask = np.any(mask_np != 0, axis=2).astype(np.int64)
        tensor = torch.from_numpy(np.ascontiguousarray(image_np.transpose(2, 0, 1)))
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return {"image": tensor, "mask": torch.from_numpy(mask), "name": sample.name}


class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, patch_grid: tuple[int, int], num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.patch_grid = patch_grid
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone.forward_features(images)
            if isinstance(features, dict):
                tokens = features.get("x_norm_patchtokens")
                if tokens is None:
                    tokens = features.get("x_prenorm")
                    if tokens is not None:
                        tokens = tokens[:, 1:, :]
            else:
                tokens = features
            if tokens is None:
                raise RuntimeError("Could not find DINOv2 patch tokens in forward_features output.")
        bsz, patches, channels = tokens.shape
        height, width = self.patch_grid
        if patches != height * width:
            side = int(round(patches ** 0.5))
            height = width = side
        x = tokens.transpose(1, 2).reshape(bsz, channels, height, width)
        logits = self.head(x)
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


def load_backbone(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, int, tuple[int, int]]:
    if args.torch_cache:
        os.environ["TORCH_HOME"] = str(args.torch_cache)
    try:
        repo_path = Path(args.dinov2_repo).expanduser()
        if repo_path.exists():
            backbone = torch.hub.load(str(repo_path), args.dinov2_model, pretrained=not args.no_pretrained, source="local")
        else:
            backbone = torch.hub.load(args.dinov2_repo, args.dinov2_model, pretrained=not args.no_pretrained)
    except Exception as exc:
        raise RuntimeError(
            "DINOv2 torch.hub load failed. Ensure the DINOv2 repo and "
            "dinov2_vitl14_reg4_pretrain.pth are cached, or run this on the "
            "Colab/AutoDL environment where loading was already verified."
        ) from exc
    backbone = backbone.to(device).eval()
    dummy = torch.zeros(1, 3, args.input_size, args.input_size, device=device)
    with torch.no_grad():
        features = backbone.forward_features(dummy)
        tokens = features["x_norm_patchtokens"] if isinstance(features, dict) else features
    _, patches, channels = tokens.shape
    side = int(round(patches ** 0.5))
    return backbone, int(channels), (side, patches // max(side, 1))


def update_confusion(confusion: np.ndarray, prediction: torch.Tensor, target: torch.Tensor) -> None:
    pred = prediction.detach().to("cpu", torch.int64).numpy().ravel()
    truth = target.detach().to("cpu", torch.int64).numpy().ravel()
    encoded = truth * 2 + pred
    confusion += np.bincount(encoded, minlength=4).reshape(2, 2)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    tn, fp = confusion[0, 0], confusion[0, 1]
    fn, tp = confusion[1, 0], confusion[1, 1]
    eps = 1e-12
    bg_iou = tn / (tn + fp + fn + eps)
    veg_iou = tp / (tp + fp + fn + eps)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    acc = (tn + tp) / (confusion.sum() + eps)
    return {"mIoU": float((bg_iou + veg_iou) / 2.0), "F1": float(f1), "Accuracy": float(acc)}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    confusion = np.zeros((2, 2), dtype=np.int64)
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, masks)
            predictions = logits.argmax(dim=1)
            update_confusion(confusion, predictions, masks)
            total_loss += float(loss.item()) * images.size(0)
            total_items += images.size(0)
    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def save_predictions(model: nn.Module, samples: list[Sample], input_size: int, device: torch.device, output_dir: Path, limit: int) -> None:
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    dataset = VegetationDataset(samples[:limit], input_size)
    model.eval()
    with torch.no_grad():
        for item in dataset:
            image = item["image"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            logits = model(image)
            pred = logits.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)
            rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            rgb[pred == 1] = VEGETATION_RGB
            Image.fromarray(rgb, mode="RGB").save(pred_dir / f"{item['name']}_pred.png")


def write_summary(output_dir: Path, results_csv: Path, config: dict[str, object], history: list[dict[str, float]]) -> None:
    summary_path = output_dir / "dinov2_smoke_summary.md"
    final = history[-1]
    lines = [
        "# DINOv2 Frozen-Backbone Smoke Train",
        "",
        "- Purpose: validate data loading, frozen DINOv2 feature extraction, lightweight segmentation-head training, mIoU calculation, and prediction-mask export.",
        f"- Dataset: `{config['dataset_root']}`.",
        f"- Train/val samples used: {config['train_samples']} / {config['val_samples']}.",
        f"- Input size: {config['input_size']}. Epochs: {config['epochs']}. Batch size: {config['batch_size']}.",
        f"- Final val loss: {final['val_loss']:.4f}. Final val mIoU: {final['val_mIoU']:.4f}. Final val F1: {final['val_F1']:.4f}.",
        f"- CSV log: `{results_csv}`.",
        "",
        "This smoke run is not a formal long-training result.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    project_root = args.project_root.resolve()
    dataset_root = (args.dataset_root or project_root / "datasets" / "2024-seg").resolve()
    output_dir = (args.output_dir or project_root / "9_DINOv2" / "smoke_outputs").resolve()
    results_csv = (args.results_csv or project_root / "results" / "dinov2_smoke_train_log.csv").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    samples = read_samples(dataset_root)
    random.Random(args.seed).shuffle(samples)
    val_count = min(max(args.max_val_samples, 1), max(1, len(samples) // 5))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Dataset: {dataset_root} ({len(samples)} total, {len(train_samples)} train, {len(val_samples)} val for smoke)")
    backbone, feature_dim, patch_grid = load_backbone(args, device)
    print(f"Loaded {args.dinov2_model}: feature_dim={feature_dim}, patch_grid={patch_grid}")

    model = DINOv2SegmentationModel(backbone, feature_dim, patch_grid).to(device)
    train_loader = DataLoader(
        VegetationDataset(train_samples, args.input_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        VegetationDataset(val_samples, args.input_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            total_items += images.size(0)
        val = evaluate(model, val_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / max(total_items, 1),
            "val_loss": val["loss"],
            "val_mIoU": val["mIoU"],
            "val_F1": val["F1"],
            "val_Accuracy": val["Accuracy"],
        }
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs}: train_loss={row['train_loss']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, val_mIoU={row['val_mIoU']:.4f}, val_F1={row['val_F1']:.4f}"
        )

    with results_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["epoch", "train_loss", "val_loss", "val_mIoU", "val_F1", "val_Accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    torch.save(model.head.state_dict(), output_dir / "dinov2_smoke_head.pth")
    save_predictions(model, val_samples, args.input_size, device, output_dir, args.save_predictions)
    config = {
        "dataset_root": str(dataset_root),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "input_size": args.input_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dinov2_model": args.dinov2_model,
        "feature_dim": feature_dim,
        "patch_grid": patch_grid,
        "elapsed_seconds": time.time() - start,
        "formal_result": False,
    }
    (output_dir / "dinov2_smoke_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_summary(output_dir, results_csv, config, history)
    print(f"Wrote smoke outputs to: {output_dir}")
    print(f"Wrote CSV log to: {results_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
