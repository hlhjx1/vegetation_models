#!/usr/bin/env python3
"""Train UNet and DeepLabV3+ on the prepared four binary vegetation datasets.

This is a new revision-stage run and does not overwrite the original eight-model
results. Zijinshan uses the temporal protocol when train.txt/test.txt exist:
2022+2023 for training and 2024 for testing. Other datasets use independent
random train/val splits unless explicit train.txt/test.txt files are present.
"""

from __future__ import annotations

import argparse
import csv
import json
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
MODEL_NAMES = ("unet", "deeplabv3plus")


@dataclass(frozen=True)
class Sample:
    name: str
    image: Path
    mask: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNet and DeepLabV3+ on four vegetation datasets.")
    parser.add_argument("--project-root", type=Path, default=Path("/content/vegetation_models_v2"))
    parser.add_argument("--binary-root", type=Path, default=Path("/content/binary"))
    parser.add_argument("--datasets", nargs="+", default=["zijinshan", "loveda", "potsdam", "vaihingen"])
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache-in-ram", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--save-predictions", type=int, default=12)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--allow-zijinshan-random-split", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch(args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def add_local_smp_path(project_root: Path) -> None:
    smp_path = project_root / "7_UNet" / "code"
    if str(smp_path) not in sys.path:
        sys.path.insert(0, str(smp_path))


def build_model(model_name: str, project_root: Path) -> nn.Module:
    add_local_smp_path(project_root)
    import segmentation_models_pytorch as smp

    if model_name == "unet":
        return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=2, activation=None)
    if model_name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=2,
            encoder_output_stride=16,
            activation=None,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def existing_file(base: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = base / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def read_split_stems(dataset_root: Path, split_name: str) -> list[str]:
    split_path = dataset_root / "ImageSets" / "Segmentation" / f"{split_name}.txt"
    if not split_path.is_file():
        return []
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def samples_from_stems(dataset_root: Path, stems: list[str], split_name: str) -> list[Sample]:
    image_dir = dataset_root / "JPEGImages"
    mask_dir = dataset_root / "SegmentationClass"
    samples = []
    missing = []
    for stem in stems:
        image_path = existing_file(image_dir, stem)
        mask_path = existing_file(mask_dir, stem)
        if image_path and mask_path:
            samples.append(Sample(stem, image_path, mask_path))
        else:
            missing.append(stem)
    if missing:
        raise FileNotFoundError(f"{dataset_root} {split_name} missing pairs: {missing[:10]}")
    if not samples:
        raise RuntimeError(f"No valid samples in {dataset_root} split {split_name}")
    return samples


def read_samples(dataset_root: Path, split_name: str = "default") -> list[Sample]:
    stems = read_split_stems(dataset_root, split_name)
    if not stems:
        raise FileNotFoundError(f"Missing or empty split file: {dataset_root / 'ImageSets' / 'Segmentation' / (split_name + '.txt')}")
    return samples_from_stems(dataset_root, stems, split_name)


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, max(1, len(shuffled) - 1))
    return shuffled[val_count:], shuffled[:val_count]


def resolve_dataset_splits(args: argparse.Namespace, dataset_name: str, dataset_root: Path) -> tuple[list[Sample], list[Sample], str, str]:
    train_stems = read_split_stems(dataset_root, "train")
    test_stems = read_split_stems(dataset_root, "test")
    if train_stems and test_stems:
        overlap = sorted(set(train_stems) & set(test_stems))
        if overlap:
            raise RuntimeError(f"{dataset_name} train/test overlap: {overlap[:10]}")
        return (
            samples_from_stems(dataset_root, train_stems, "train"),
            samples_from_stems(dataset_root, test_stems, "test"),
            "explicit_train_test",
            "test",
        )
    if dataset_name == "zijinshan" and not args.allow_zijinshan_random_split:
        raise FileNotFoundError("Zijinshan must use train.txt/test.txt. Run prepare_zijinshan_temporal_colab.py first.")
    samples = read_samples(dataset_root, "default")
    train_samples, eval_samples = split_samples(samples, args.val_ratio, args.seed)
    return train_samples, eval_samples, "random_train_val", "val"


class VegetationDataset(Dataset):
    def __init__(self, samples: list[Sample], input_size: int, augment: bool, seed: int, cache_in_ram: bool = False) -> None:
        self.samples = samples
        self.input_size = input_size
        self.augment = augment
        self.seed = seed
        self.cache_in_ram = cache_in_ram
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        if self.cache_in_ram and index in self._cache:
            image_tensor, mask_tensor = self._cache[index]
            return {"image": image_tensor, "mask": mask_tensor, "name": sample.name}
        image = Image.open(sample.image).convert("RGB")
        mask_rgb = Image.open(sample.mask).convert("RGB")
        if self.augment:
            rng = random.Random(self.seed + index)
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask_rgb = mask_rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask_rgb = mask_rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image = image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        mask_rgb = mask_rgb.resize((self.input_size, self.input_size), Image.Resampling.NEAREST)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask_rgb, dtype=np.uint8)
        mask = np.any(mask_np != 0, axis=2).astype(np.int64)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_np.transpose(2, 0, 1)))
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        mask_tensor = torch.from_numpy(mask)
        if self.cache_in_ram:
            self._cache[index] = (image_tensor, mask_tensor)
        return {"image": image_tensor, "mask": mask_tensor, "name": sample.name}


def amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def logits_from_model_output(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["out"]
    return output


def update_confusion(confusion: np.ndarray, prediction: torch.Tensor, target: torch.Tensor) -> None:
    pred = prediction.detach().to("cpu", torch.int64).numpy().ravel()
    truth = target.detach().to("cpu", torch.int64).numpy().ravel()
    confusion += np.bincount(truth * 2 + pred, minlength=4).reshape(2, 2)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    tn, fp = confusion[0, 0], confusion[0, 1]
    fn, tp = confusion[1, 0], confusion[1, 1]
    eps = 1e-12
    bg_iou = tn / (tn + fp + fn + eps)
    veg_iou = tp / (tp + fp + fn + eps)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    acc = (tn + tp) / (confusion.sum() + eps)
    return {"mIoU": float((bg_iou + veg_iou) / 2.0), "veg_IoU": float(veg_iou), "F1": float(f1), "Accuracy": float(acc)}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool, channels_last: bool) -> dict[str, float]:
    model.eval()
    confusion = np.zeros((2, 2), dtype=np.int64)
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            if channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            masks = batch["mask"].to(device)
            with amp_context(device, use_amp):
                logits = logits_from_model_output(model(images))
                loss = F.cross_entropy(logits, masks)
            update_confusion(confusion, logits.argmax(dim=1), masks)
            total_loss += float(loss.item()) * images.size(0)
            total_items += images.size(0)
    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def save_predictions(model: nn.Module, samples: list[Sample], input_size: int, device: torch.device, output_dir: Path, limit: int, channels_last: bool) -> None:
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    dataset = VegetationDataset(samples[:limit], input_size, augment=False, seed=0)
    model.eval()
    with torch.no_grad():
        for item in dataset:
            image = item["image"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            if channels_last and device.type == "cuda":
                image = image.contiguous(memory_format=torch.channels_last)
            pred = logits_from_model_output(model(image)).argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)
            rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            rgb[pred == 1] = VEGETATION_RGB
            Image.fromarray(rgb, mode="RGB").save(pred_dir / f"{item['name']}_pred.png")


def train_one(args: argparse.Namespace, model_name: str, dataset_name: str, device: torch.device) -> dict[str, object]:
    dataset_root = args.binary_root / dataset_name
    train_samples, eval_samples, split_protocol, eval_split_name = resolve_dataset_splits(args, dataset_name, dataset_root)
    output_dir = (args.output_root / model_name / dataset_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_name, args.project_root).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        VegetationDataset(train_samples, args.input_size, augment=True, seed=args.seed, cache_in_ram=args.cache_in_ram),
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        VegetationDataset(eval_samples, args.input_size, augment=False, seed=args.seed, cache_in_ram=args.cache_in_ram),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    history: list[dict[str, float | int | str]] = []
    best_score = -1.0
    best_epoch = 0
    bad_epochs = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            if args.channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.amp):
                logits = logits_from_model_output(model(images))
                loss = F.cross_entropy(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            total_items += images.size(0)
        scheduler.step()
        metrics = evaluate(model, eval_loader, device, args.amp, args.channels_last)
        row = {
            "model": model_name,
            "dataset": dataset_name,
            "epoch": epoch,
            "train_loss": total_loss / max(total_items, 1),
            f"{eval_split_name}_loss": metrics["loss"],
            f"{eval_split_name}_mIoU": metrics["mIoU"],
            f"{eval_split_name}_veg_IoU": metrics["veg_IoU"],
            f"{eval_split_name}_F1": metrics["F1"],
            f"{eval_split_name}_Accuracy": metrics["Accuracy"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"[{model_name}/{dataset_name}] epoch {epoch}/{args.epochs}: "
            f"train_loss={row['train_loss']:.4f}, {eval_split_name}_mIoU={row[f'{eval_split_name}_mIoU']:.4f}, "
            f"{eval_split_name}_veg_IoU={row[f'{eval_split_name}_veg_IoU']:.4f}, {eval_split_name}_F1={row[f'{eval_split_name}_F1']:.4f}"
        )
        score = float(row[f"{eval_split_name}_mIoU"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            bad_epochs = 0
        else:
            bad_epochs += 1
        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"[{model_name}/{dataset_name}] early stop after {bad_epochs} non-improving epochs.")
            break

    with (output_dir / "train_log.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))
    save_predictions(model, eval_samples, args.input_size, device, output_dir, args.save_predictions, args.channels_last)
    best = max(history, key=lambda item: float(item[f"{eval_split_name}_mIoU"]))
    config = {
        "model": model_name,
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "split_protocol": split_protocol,
        "eval_split_name": eval_split_name,
        "total_samples": len(train_samples) + len(eval_samples),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "input_size": args.input_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "best_epoch": best_epoch,
        "best_eval_mIoU": best_score,
        "elapsed_seconds": time.time() - start,
        "amp": bool(args.amp),
        "cache_in_ram": bool(args.cache_in_ram),
        "channels_last": bool(args.channels_last),
        "allow_tf32": bool(args.allow_tf32),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {
        **config,
        "best_eval_loss": best[f"{eval_split_name}_loss"],
        "best_eval_veg_IoU": best[f"{eval_split_name}_veg_IoU"],
        "best_eval_F1": best[f"{eval_split_name}_F1"],
        "best_eval_Accuracy": best[f"{eval_split_name}_Accuracy"],
    }


def main() -> int:
    args = parse_args()
    if args.output_root is None:
        args.output_root = args.project_root / "13_ClassicBaselines" / "four_dataset_runs"
    if args.results_csv is None:
        args.results_csv = args.project_root / "results" / "classic_two_four_dataset_results.csv"
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    configure_torch(args)
    device = torch.device(args.device)
    print(f"Device: {device}")
    summaries = []
    for model_name in args.models:
        for dataset_name in args.datasets:
            summaries.append(train_one(args, model_name, dataset_name, device))
    fieldnames = sorted({key for row in summaries for key in row.keys()})
    with args.results_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote aggregate CSV: {args.results_csv}")
    print(f"Wrote per-run outputs: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
