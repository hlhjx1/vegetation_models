#!/usr/bin/env python3
"""Colab runner for the four added model efficiency and robustness experiments.

This script is intentionally separate from the old eight-model scripts. It
does not overwrite the old benchmark CSV files. Default outputs:

- results/four_model_efficiency_colab.csv
- results/four_model_robustness_colab.csv
- results/four_model_colab_path_report.json

Typical Colab setup:

    from google.colab import drive
    drive.mount("/content/drive")

    !ln -sfn /content/drive/MyDrive/vegetation_models_v2 /content/vegetation_models_v2
    %cd /content/vegetation_models_v2

    !pip install -q -U transformers accelerate qwen-vl-utils thop
    # LocateAnything may require its pinned stack:
    # !pip install -q opencv-python-headless==4.11.0.86 transformers==4.57.1 peft torchvision decord==0.6.0 lmdb==1.7.5 accelerate

Preflight only:

    !python scripts/colab_four_model_efficiency_robustness.py --mode preflight

DINOv2/DINOv3 efficiency and robustness:

    !python scripts/colab_four_model_efficiency_robustness.py --mode all_dino --datasets zijinshan loveda potsdam vaihingen

Prompt-model latency/robustness, slower and memory-heavy:

    !python scripts/colab_four_model_efficiency_robustness.py --mode prompt --datasets zijinshan loveda potsdam vaihingen --rounds-per-dataset 100

Full run:

    !python scripts/colab_four_model_efficiency_robustness.py --mode all --datasets zijinshan loveda potsdam vaihingen --rounds-per-dataset 100

Important:
- DINOv2 and DINOv3 require the local trained heads:
  9_DINOv2/four_dataset_runs/<dataset>/best_head.pth
  10_DINOv3/four_dataset_runs/<dataset>/best_head.pth
- DINOv3 also requires the local DINOv3 repo and SAT-493M weight file.
- QwenVL2.5-3B and LocateAnything are prompt-based/grounding models; their
  latency is prompt-to-box generation latency and should not be mixed with
  dense segmentation forward-only latency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset


DATASETS = ("zijinshan", "loveda", "potsdam", "vaihingen")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
DINO_INPUT_SIZE = {"dinov2": 518, "dinov3": 512}
DINO_MEAN_STD = {
    "dinov2": (
        torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
        torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
    ),
    "dinov3": (
        torch.tensor([0.430, 0.411, 0.296]).view(3, 1, 1),
        torch.tensor([0.213, 0.156, 0.143]).view(3, 1, 1),
    ),
}


@dataclass(frozen=True)
class Sample:
    dataset: str
    split: str
    name: str
    image: Path
    mask: Path
    round_index: int


@dataclass(frozen=True)
class Condition:
    perturbation: str
    strength: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab four-model efficiency and robustness runner.")
    parser.add_argument("--project-root", type=Path, default=Path("/content/vegetation_models_v2"))
    parser.add_argument("--binary-root", type=Path, default=Path("/content/binary"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--mode",
        choices=["preflight", "dino_efficiency", "dino_robustness", "all_dino", "prompt", "all"],
        default="preflight",
    )
    parser.add_argument("--split", choices=["auto", "test", "val", "default"], default="auto")
    parser.add_argument("--rounds-per-dataset", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for DINO inference.")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--no-flops", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--locate-model-id", default="nvidia/LocateAnything-3B")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-locateanything", action="store_true")
    parser.add_argument("--efficiency-csv", type=Path, default=None)
    parser.add_argument("--robustness-csv", type=Path, default=None)
    parser.add_argument("--path-report-json", type=Path, default=None)
    parser.add_argument("--perturbed-image-root", type=Path, default=None)
    return parser.parse_args()


def configure_torch(args: argparse.Namespace) -> None:
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def existing_file(base: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = base / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def read_split(dataset_root: Path, split_name: str) -> list[str]:
    path = dataset_root / "ImageSets" / "Segmentation" / f"{split_name}.txt"
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_split(dataset_root: Path, requested: str) -> tuple[str, list[str]]:
    if requested != "auto":
        stems = read_split(dataset_root, requested)
        if not stems:
            raise FileNotFoundError(f"Missing split {requested}.txt under {dataset_root}")
        return requested, stems
    for split_name in ("test", "val", "default"):
        stems = read_split(dataset_root, split_name)
        if stems:
            return split_name, stems
    raise FileNotFoundError(f"No usable split file under {dataset_root}")


def collect_samples(binary_root: Path, datasets: list[str], split: str, rounds: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    all_samples: list[Sample] = []
    for dataset in datasets:
        root = binary_root / dataset
        split_name, stems = resolve_split(root, split)
        valid: list[tuple[str, Path, Path]] = []
        for stem in stems:
            image = existing_file(root / "JPEGImages", stem)
            mask = existing_file(root / "SegmentationClass", stem)
            if image and mask:
                valid.append((stem, image, mask))
        if not valid:
            raise FileNotFoundError(f"No valid image/mask pairs for {dataset} under {root}")
        chosen = valid[:]
        rng.shuffle(chosen)
        if len(chosen) >= rounds:
            chosen = chosen[:rounds]
        else:
            chosen = [rng.choice(valid) for _ in range(rounds)]
        for idx, (stem, image, mask) in enumerate(chosen, start=1):
            all_samples.append(Sample(dataset, split_name, stem, image, mask, idx))
        print(f"{dataset}: split={split_name}, valid={len(valid)}, scheduled={len(chosen)}")
    return all_samples


def all_eval_samples(binary_root: Path, datasets: list[str], split: str) -> list[Sample]:
    samples: list[Sample] = []
    for dataset in datasets:
        root = binary_root / dataset
        split_name, stems = resolve_split(root, split)
        idx = 0
        for stem in stems:
            image = existing_file(root / "JPEGImages", stem)
            mask = existing_file(root / "SegmentationClass", stem)
            if image and mask:
                idx += 1
                samples.append(Sample(dataset, split_name, stem, image, mask, idx))
        print(f"{dataset}: split={split_name}, eval_pairs={idx}")
    return samples


def perturb_image(img: Image.Image, condition: Condition, identifier: str = "") -> Image.Image:
    img = img.convert("RGB")
    if condition.perturbation == "clean":
        return img
    if condition.perturbation == "brightness":
        return ImageEnhance.Brightness(img).enhance(float(condition.strength))
    if condition.perturbation == "contrast":
        return ImageEnhance.Contrast(img).enhance(float(condition.strength))
    if condition.perturbation == "gaussian_blur":
        kernel = int(condition.strength.replace("kernel_", ""))
        try:
            import cv2

            image = np.asarray(img, dtype=np.uint8)
            altered = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REFLECT_101)
            return Image.fromarray(altered.astype(np.uint8), mode="RGB")
        except Exception:
            return img.filter(ImageFilter.GaussianBlur(radius=max(1.0, (kernel - 1) / 2.0)))
    if condition.perturbation == "shadow_illumination":
        settings = {
            "mild": (0.82, 0.34, 0.12),
            "moderate": (0.64, 0.52, 0.16),
            "strong": (0.46, 0.68, 0.20),
        }
        minimum, coverage, feather = settings[condition.strength]
        image = np.asarray(img, dtype=np.uint8)
        height, width = image.shape[:2]
        digest = hashlib.sha256(identifier.encode("utf-8")).digest()
        angle = (int.from_bytes(digest[:2], "little") / 65535.0 - 0.5) * 0.85
        offset = (int.from_bytes(digest[2:4], "little") / 65535.0 - 0.5) * 0.35
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        x = xx / max(width - 1, 1) - 0.5
        y = yy / max(height - 1, 1) - 0.5
        signed_distance = x * np.cos(angle) + y * np.sin(angle) - offset
        transition = max(feather, 1e-4)
        field = 1.0 / (1.0 + np.exp((signed_distance + coverage / 2.0) / transition))
        multiplier = 1.0 - (1.0 - minimum) * field
        altered = np.clip(image.astype(np.float32) * multiplier[..., None], 0, 255).astype(np.uint8)
        return Image.fromarray(altered, mode="RGB")
    if condition.perturbation == "gaussian_noise":
        digest = hashlib.sha256(identifier.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0.0, float(condition.strength), arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
    raise ValueError(f"Unknown perturbation: {condition}")


def conditions() -> list[Condition]:
    return [
        Condition("clean", "none"),
        Condition("brightness", "0.70"),
        Condition("brightness", "0.85"),
        Condition("brightness", "1.15"),
        Condition("brightness", "1.30"),
        Condition("contrast", "0.75"),
        Condition("contrast", "0.90"),
        Condition("contrast", "1.10"),
        Condition("contrast", "1.25"),
        Condition("gaussian_blur", "kernel_3"),
        Condition("gaussian_blur", "kernel_5"),
        Condition("gaussian_blur", "kernel_7"),
        Condition("shadow_illumination", "mild"),
        Condition("shadow_illumination", "moderate"),
        Condition("shadow_illumination", "strong"),
    ]


def scan_paths(project_root: Path) -> dict[str, dict[str, list[str]]]:
    candidates = {
        "project_root": [project_root, Path("/content/drive/MyDrive/vegetation_models_v2")],
        "binary_root": [Path("/content/binary"), project_root / "datasets_binary"],
        "dinov2_code_or_cache": [
            project_root / "9_DINOv2",
            Path("/root/.cache/torch/hub/facebookresearch_dinov2_main"),
            Path("/content/vegetation_foundation_models/_cache/torch_cache/hub/facebookresearch_dinov2_main"),
        ],
        "dinov2_pretrain": [
            Path("/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"),
            project_root / "_cache/torch_cache/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth",
            Path("/content/vegetation_foundation_models/_cache/torch_cache/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"),
        ],
        "dinov2_heads": [project_root / "9_DINOv2/four_dataset_runs" / d / "best_head.pth" for d in DATASETS],
        "dinov3_repo": [
            project_root / "10_DINOv3/code/dinov3-main",
            Path("/content/drive/MyDrive/vegetation_models_v2/10_DINOv3/code/dinov3-main"),
        ],
        "dinov3_pretrain": [
            project_root / "10_DINOv3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
            Path("/content/drive/MyDrive/vegetation_models_v2/10_DINOv3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
        ],
        "dinov3_heads": [project_root / "10_DINOv3/four_dataset_runs" / d / "best_head.pth" for d in DATASETS],
        "qwen_weights_or_cache": [
            project_root / "11_QwenVL/weights",
            Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"),
        ],
        "locateanything_weights_or_cache": [
            project_root / "12_LocatingAnything/weights",
            Path("/root/.cache/huggingface/hub/models--nvidia--LocateAnything-3B"),
        ],
    }
    report = {}
    for key, paths in candidates.items():
        report[key] = {
            "found": [str(p) for p in paths if p.exists()],
            "checked": [str(p) for p in paths],
        }
    return report


class PerturbedDinoDataset(Dataset):
    def __init__(self, samples: list[Sample], input_size: int, condition: Condition, mean: torch.Tensor, std: torch.Tensor):
        self.samples = samples
        self.input_size = input_size
        self.condition = condition
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = Image.open(sample.image).convert("RGB")
        image = perturb_image(image, self.condition, identifier=sample.image.name)
        mask_rgb = Image.open(sample.mask).convert("RGB")
        image = image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        mask_rgb = mask_rgb.resize((self.input_size, self.input_size), Image.Resampling.NEAREST)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask_rgb, dtype=np.uint8)
        mask = np.any(mask_np != 0, axis=2).astype(np.int64)
        tensor = torch.from_numpy(np.ascontiguousarray(image_np.transpose(2, 0, 1)))
        tensor = (tensor - self.mean) / self.std
        return {"image": tensor, "mask": torch.from_numpy(mask), "name": sample.name}


def confusion_metrics(confusion: np.ndarray) -> dict[str, float]:
    tn, fp = confusion[0, 0], confusion[0, 1]
    fn, tp = confusion[1, 0], confusion[1, 1]
    eps = 1e-12
    bg_iou = tn / (tn + fp + fn + eps)
    veg_iou = tp / (tp + fp + fn + eps)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    acc = (tn + tp) / (confusion.sum() + eps)
    return {"mIoU": float((bg_iou + veg_iou) / 2.0), "veg_IoU": float(veg_iou), "F1": float(f1), "Accuracy": float(acc)}


def update_confusion(confusion: np.ndarray, pred: torch.Tensor, target: torch.Tensor) -> None:
    pred_np = pred.detach().to("cpu", torch.int64).numpy().ravel()
    truth = target.detach().to("cpu", torch.int64).numpy().ravel()
    confusion += np.bincount(truth * 2 + pred_np, minlength=4).reshape(2, 2)


def amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def load_dino_model(kind: str, dataset: str, args: argparse.Namespace):
    sys.path.insert(0, str(args.project_root / "scripts"))
    base = import_module(args.project_root / "scripts/run_dinov2_four_dataset_train.py", "colab_dinov2_base")
    device = torch.device(args.device)
    if kind == "dinov2":
        local_dinov2_repos = [
            args.project_root / "9_DINOv2/code/dinov2",
            Path("/content/vegetation_foundation_models/code/dinov2"),
            Path("/content/vegetation_foundation_models/code/dinov2-main"),
            Path("/content/vegetation_foundation_models/code/dinov2-main-main"),
        ]
        dinov2_repo = "facebookresearch/dinov2"
        for candidate in local_dinov2_repos:
            if (candidate / "hubconf.py").is_file():
                dinov2_repo = str(candidate)
                break
        ns = argparse.Namespace(
            torch_cache=None,
            dinov2_repo=dinov2_repo,
            dinov2_model="dinov2_vitl14_reg",
            input_size=DINO_INPUT_SIZE[kind],
            no_pretrained=False,
        )
        backbone, feature_dim, patch_grid = base.load_backbone(ns, device)
        head_path = args.project_root / "9_DINOv2/four_dataset_runs" / dataset / "best_head.pth"
    elif kind == "dinov3":
        d3 = import_module(args.project_root / "scripts/run_dinov3_four_dataset_train.py", "colab_dinov3_runner")
        repo_candidates = [
            args.project_root / "10_DINOv3/code/dinov3-main",
            Path("/content/vegetation_foundation_models/code/dinov3"),
            Path("/content/vegetation_foundation_models/code/dinov3-main"),
        ]
        weight_candidates = [
            args.project_root / "10_DINOv3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
            Path("/content/vegetation_foundation_models/code/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
            Path("/content/vegetation_foundation_models/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
            Path("/content/vegetation_foundation_models/code/dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
        ]
        repo = next((path for path in repo_candidates if path.exists()), repo_candidates[0])
        weights = next((path for path in weight_candidates if path.is_file()), weight_candidates[0])
        ns = argparse.Namespace(
            torch_cache=None,
            dinov3_repo=repo,
            dinov3_model="dinov3_vitl16",
            dinov3_weights=weights,
            input_size=DINO_INPUT_SIZE[kind],
            no_pretrained=False,
        )
        backbone, feature_dim, patch_grid = d3.load_backbone(ns, device)
        head_path = args.project_root / "10_DINOv3/four_dataset_runs" / dataset / "best_head.pth"
    else:
        raise ValueError(kind)
    if not head_path.is_file():
        raise FileNotFoundError(f"Missing trained head: {head_path}")
    model = base.DINOv2SegmentationModel(backbone, feature_dim, patch_grid).to(device)
    model.head.load_state_dict(torch.load(head_path, map_location=device))
    model.eval()
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model, head_path


def model_param_count(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_flops_g(model: torch.nn.Module, sample: torch.Tensor) -> tuple[str, str]:
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(sample,), verbose=False)
        return f"{2.0 * float(macs) / 1e9:.3f}", "2x THOP MACs"
    except Exception as exc:
        return "pending", f"THOP failed: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"


def benchmark_model(model: torch.nn.Module, input_size: int, args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    sample = torch.zeros(1, 3, input_size, input_size, device=device)
    if args.channels_last and device.type == "cuda":
        sample = sample.contiguous(memory_format=torch.channels_last)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(args.warmup):
            with amp_context(device, args.amp):
                _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.repeats):
            with amp_context(device, args.amp):
                _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000.0 / max(args.repeats, 1)
    params, trainable = model_param_count(model)
    if args.no_flops:
        flops, flops_note = "pending", "Disabled by --no-flops"
    else:
        flops, flops_note = model_flops_g(model, sample)
    memory = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    return {
        "params_M": params / 1e6,
        "trainable_params_M": trainable / 1e6,
        "flops_G": flops,
        "flops_note": flops_note,
        "inference_time_ms": ms,
        "fps": 1000.0 / ms if ms > 0 else 0.0,
        "gpu_memory_MB": memory,
    }


def run_dino_efficiency(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind, model_name in [("dinov2", "DINOv2 ViT-L/14 reg + head"), ("dinov3", "DINOv3 ViT-L/16 SAT-493M + head")]:
        for dataset in args.datasets:
            print(f"[efficiency] loading {kind}/{dataset}")
            model, head_path = load_dino_model(kind, dataset, args)
            stats = benchmark_model(model, DINO_INPUT_SIZE[kind], args)
            rows.append(
                {
                    "model": model_name,
                    "dataset_head": dataset,
                    "task_type": "frozen-head segmentation",
                    "input_size": f"{DINO_INPUT_SIZE[kind]}x{DINO_INPUT_SIZE[kind]}",
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device,
                    "head_path": str(head_path),
                    **stats,
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def evaluate_dino_condition(
    model: torch.nn.Module,
    samples: list[Sample],
    kind: str,
    condition: Condition,
    args: argparse.Namespace,
) -> dict[str, float]:
    mean, std = DINO_MEAN_STD[kind]
    dataset = PerturbedDinoDataset(samples, DINO_INPUT_SIZE[kind], condition, mean, std)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.device == "cuda")
    confusion = np.zeros((2, 2), dtype=np.int64)
    device = torch.device(args.device)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            if args.channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            masks = batch["mask"].to(device)
            with amp_context(device, args.amp):
                logits = model(images)
            update_confusion(confusion, logits.argmax(dim=1), masks)
    return confusion_metrics(confusion)


def run_dino_robustness(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    eval_samples = all_eval_samples(args.binary_root, args.datasets, args.split)
    by_dataset = {dataset: [s for s in eval_samples if s.dataset == dataset] for dataset in args.datasets}
    for kind, model_name in [("dinov2", "DINOv2 ViT-L/14 reg + head"), ("dinov3", "DINOv3 ViT-L/16 SAT-493M + head")]:
        for dataset in args.datasets:
            print(f"[robustness] loading {kind}/{dataset}")
            model, head_path = load_dino_model(kind, dataset, args)
            clean_metrics: dict[str, float] | None = None
            for condition in conditions():
                print(f"[robustness] {kind}/{dataset}: {condition.perturbation}:{condition.strength}")
                metrics = evaluate_dino_condition(model, by_dataset[dataset], kind, condition, args)
                if condition.perturbation == "clean":
                    clean_metrics = metrics
                assert clean_metrics is not None
                rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "task_type": "frozen-head segmentation",
                        "metric_type": "dense segmentation",
                        "perturbation": condition.perturbation,
                        "strength": condition.strength,
                        "n": len(by_dataset[dataset]),
                        "mIoU": metrics["mIoU"],
                        "veg_IoU": metrics["veg_IoU"],
                        "F1": metrics["F1"],
                        "Accuracy": metrics["Accuracy"],
                        "Delta_mIoU": metrics["mIoU"] - clean_metrics["mIoU"],
                        "Delta_F1": metrics["F1"] - clean_metrics["F1"],
                        "Delta_Accuracy": metrics["Accuracy"] - clean_metrics["Accuracy"],
                        "head_path": str(head_path),
                    }
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def save_perturbed_temp(sample: Sample, condition: Condition, root: Path) -> Path:
    out_dir = root / sample.dataset / condition.perturbation / condition.strength
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{sample.name}.png"
    if not out.exists():
        img = Image.open(sample.image).convert("RGB")
        perturbed = perturb_image(img, condition, identifier=sample.image.name)
        perturbed.save(out)
    return out


def summarize_box_rows(rows: list[dict[str, object]], clean_lookup: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["dataset"]), str(row["perturbation"]), str(row["strength"]))
        grouped.setdefault(key, []).append(row)
    for (model, dataset, perturbation, strength), group in grouped.items():
        metrics = {
            "box_mask_iou": float(np.mean([float(r["box_mask_iou"]) for r in group])),
            "box_mask_f1": float(np.mean([float(r["box_mask_f1"]) for r in group])),
            "box_mask_accuracy": float(np.mean([float(r["box_mask_accuracy"]) for r in group])),
            "avg_boxes": float(np.mean([float(r["box_count"]) for r in group])),
        }
        clean = clean_lookup.get((model, dataset), metrics)
        delta_iou = metrics["box_mask_iou"] - clean["box_mask_iou"]
        delta_f1 = metrics["box_mask_f1"] - clean["box_mask_f1"]
        delta_accuracy = metrics["box_mask_accuracy"] - clean["box_mask_accuracy"]
        summary.append(
            {
                "model": model,
                "dataset": dataset,
                "task_type": "prompt-based inference" if "Qwen" in model else "grounding",
                "metric_type": "box-mask",
                "perturbation": perturbation,
                "strength": strength,
                "n": len(group),
                "mIoU": metrics["box_mask_iou"],
                "F1": metrics["box_mask_f1"],
                "Accuracy": metrics["box_mask_accuracy"],
                "Delta_mIoU": delta_iou,
                "Delta_F1": delta_f1,
                "Delta_Accuracy": delta_accuracy,
                "box_mask_iou": metrics["box_mask_iou"],
                "box_mask_f1": metrics["box_mask_f1"],
                "box_mask_accuracy": metrics["box_mask_accuracy"],
                "avg_boxes": metrics["avg_boxes"],
                "Delta_IoU": delta_iou,
            }
        )
    return summary


def run_qwen_prompt(args: argparse.Namespace, temp_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    qwen = import_module(args.project_root / "scripts/run_qwenvl_vegetation_prompt_eval.py", "colab_qwen_eval")
    ns = argparse.Namespace(model_id=args.qwen_model_id, torch_dtype="auto", device=args.device)
    model, processor, process_vision_info = qwen.load_qwen(ns)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    samples = collect_samples(args.binary_root, args.datasets, args.split, args.rounds_per_dataset, args.seed)
    detail_rows: list[dict[str, object]] = []
    clean_lookup: dict[tuple[str, str], dict[str, float]] = {}
    for condition in conditions():
        for idx, sample in enumerate(samples, start=1):
            image_path = sample.image if condition.perturbation == "clean" else save_perturbed_temp(sample, condition, temp_root)
            image = Image.open(image_path).convert("RGB")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            raw = qwen.run_one(model, processor, process_vision_info, image_path, args.max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000.0
            parsed = qwen.parse_json_response(raw)
            boxes = parsed.get("boxes", [])
            pred_mask, modes = qwen.boxes_to_mask(boxes if isinstance(boxes, list) else [], image.size, "auto")
            metrics = qwen.mask_metrics(pred_mask, sample.mask, image.size)
            row = {
                "model": args.qwen_model_id,
                "dataset": sample.dataset,
                "sample": sample.name,
                "round_index": sample.round_index,
                "perturbation": condition.perturbation,
                "strength": condition.strength,
                "box_count": len(boxes) if isinstance(boxes, list) else 0,
                "pixel_box_count": modes["pixel"],
                "normalized1000_box_count": modes["normalized1000"],
                "latency_ms": latency_ms,
                "params_M": params_m,
                **metrics,
            }
            detail_rows.append(row)
            print(f"[qwen] {condition.perturbation}:{condition.strength} {idx}/{len(samples)} {sample.dataset}/{sample.name} IoU={metrics['box_mask_iou']:.4f}")
    summary = summarize_box_rows(detail_rows, clean_lookup={})
    for row in summary:
        if row["perturbation"] == "clean":
            clean_lookup[(str(row["model"]), str(row["dataset"]))] = {
                "box_mask_iou": float(row["box_mask_iou"]),
                "box_mask_f1": float(row["box_mask_f1"]),
                "box_mask_accuracy": float(row["box_mask_accuracy"]),
            }
    summary = summarize_box_rows(detail_rows, clean_lookup)
    return summary, detail_rows


def run_locate_prompt(args: argparse.Namespace, temp_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    locate = import_module(args.project_root / "scripts/run_locateanything_vegetation_grounding.py", "colab_locate_eval")
    model, processor, tokenizer = locate.load_model(args.locate_model_id, args.device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    samples = collect_samples(args.binary_root, args.datasets, args.split, args.rounds_per_dataset, args.seed)
    detail_rows: list[dict[str, object]] = []
    for condition in conditions():
        for idx, sample in enumerate(samples, start=1):
            image_path = sample.image if condition.perturbation == "clean" else save_perturbed_temp(sample, condition, temp_root)
            image = Image.open(image_path).convert("RGB")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            raw = locate.run_one(model, processor, tokenizer, image_path, args.max_new_tokens, args.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000.0
            parsed = locate.parse_json_response(raw)
            boxes = parsed.get("boxes", [])
            pred_mask = locate.boxes_to_mask(boxes if isinstance(boxes, list) else [], image.size)
            metrics = locate.mask_metrics(pred_mask, sample.mask, image.size)
            row = {
                "model": args.locate_model_id,
                "dataset": sample.dataset,
                "sample": sample.name,
                "round_index": sample.round_index,
                "perturbation": condition.perturbation,
                "strength": condition.strength,
                "box_count": len(boxes) if isinstance(boxes, list) else 0,
                "latency_ms": latency_ms,
                "params_M": params_m,
                **metrics,
            }
            detail_rows.append(row)
            print(f"[locate] {condition.perturbation}:{condition.strength} {idx}/{len(samples)} {sample.dataset}/{sample.name} IoU={metrics['box_mask_iou']:.4f}")
    clean_lookup: dict[tuple[str, str], dict[str, float]] = {}
    first_summary = summarize_box_rows(detail_rows, clean_lookup={})
    for row in first_summary:
        if row["perturbation"] == "clean":
            clean_lookup[(str(row["model"]), str(row["dataset"]))] = {
                "box_mask_iou": float(row["box_mask_iou"]),
                "box_mask_f1": float(row["box_mask_f1"]),
                "box_mask_accuracy": float(row["box_mask_accuracy"]),
            }
    return summarize_box_rows(detail_rows, clean_lookup), detail_rows


def run_prompt(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    temp_root = args.perturbed_image_root or (args.project_root / "results/four_model_perturbed_prompt_images")
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    if not args.skip_qwen:
        qwen_summary, qwen_detail = run_qwen_prompt(args, temp_root)
        summary_rows.extend(qwen_summary)
        detail_rows.extend(qwen_detail)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not args.skip_locateanything:
        locate_summary, locate_detail = run_locate_prompt(args, temp_root)
        summary_rows.extend(locate_summary)
        detail_rows.extend(locate_detail)
    return summary_rows, detail_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def append_or_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    write_csv(path, rows)


def main() -> int:
    args = parse_args()
    configure_torch(args)
    results_dir = args.project_root / "results"
    args.efficiency_csv = args.efficiency_csv or (results_dir / "four_model_efficiency_colab.csv")
    args.robustness_csv = args.robustness_csv or (results_dir / "four_model_robustness_colab.csv")
    args.path_report_json = args.path_report_json or (results_dir / "four_model_colab_path_report.json")
    results_dir.mkdir(parents=True, exist_ok=True)

    report = scan_paths(args.project_root)
    args.path_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote path report: {args.path_report_json}")
    if args.mode == "preflight":
        return 0

    efficiency_rows: list[dict[str, object]] = []
    robustness_rows: list[dict[str, object]] = []

    if args.mode in ("dino_efficiency", "all_dino", "all"):
        efficiency_rows.extend(run_dino_efficiency(args))
        append_or_write_csv(args.efficiency_csv, efficiency_rows)

    if args.mode in ("dino_robustness", "all_dino", "all"):
        robustness_rows.extend(run_dino_robustness(args))
        append_or_write_csv(args.robustness_csv, robustness_rows)

    if args.mode in ("prompt", "all"):
        prompt_summary, prompt_detail = run_prompt(args)
        robustness_rows.extend(prompt_summary)
        append_or_write_csv(args.robustness_csv, robustness_rows)
        detail_path = results_dir / "four_model_prompt_detail_colab.csv"
        write_csv(detail_path, prompt_detail)
        prompt_eff_rows = []
        for key, group in group_by(prompt_detail, ["model", "dataset"]).items():
            model, dataset = key
            clean = [r for r in group if r["perturbation"] == "clean"]
            values = clean or group
            prompt_eff_rows.append(
                {
                    "model": model,
                    "dataset_head": dataset,
                    "task_type": "prompt-based inference" if "Qwen" in str(model) else "grounding",
                    "params_M": float(values[0].get("params_M", 0.0)),
                    "flops_G": "not_comparable_generation_dependent",
                    "inference_time_ms": float(np.mean([float(r["latency_ms"]) for r in values])),
                    "fps": 1000.0 / float(np.mean([float(r["latency_ms"]) for r in values])),
                    "input_size": "original image",
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device,
                    "notes": "Prompt-to-box generation latency; not dense segmentation forward-only latency.",
                }
            )
        efficiency_rows.extend(prompt_eff_rows)
        append_or_write_csv(args.efficiency_csv, efficiency_rows)

    return 0


def group_by(rows: list[dict[str, object]], keys: list[str]) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    return grouped


if __name__ == "__main__":
    raise SystemExit(main())
