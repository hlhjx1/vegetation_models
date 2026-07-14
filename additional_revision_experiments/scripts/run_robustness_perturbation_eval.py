"""Inference-only robustness evaluation for the vegetation segmentation study.

This script addresses Reviewer 2, Comment 4 without retraining.  It applies
the same deterministic image perturbations to the held-out 2024 test images,
runs the six models with valid fine-tuned checkpoints, and calculates all
metrics with the same pixel-level implementation.

Protocol
--------
* Dataset: datasets/2024-seg (JPEGImages + SegmentationClass), never training data.
* Models: SAM2-Tiny, MobileSAM, MobileSAMV2, YOLO11s-seg, UNet, DeepLabV3+.
* Metrics: two-class mean IoU (background and vegetation), vegetation F1, and
  pixel accuracy, accumulated globally over the identical image subset.
* Perturbations: brightness, contrast, Gaussian blur, and deterministic
  synthetic cast shadows.  Ground-truth masks are never perturbed.

The default formal protocol evaluates a deterministic evenly spaced 32-image
subset of the 67-image 2024 held-out set.  Use --max-samples 0 for all 67
images.  A manifest containing the exact selected filenames is written with
the results so the experiment is auditable and repeatable.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import pkgutil
import sys
import traceback
import zipimport
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

# Python 3.12 compatibility for older Ultralytics dependency combinations.
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = zipimport.zipimporter  # type: ignore[attr-defined]
if not hasattr(importlib.machinery.FileFinder, "find_module"):
    def _file_finder_find_module(self, fullname: str, path: str | None = None):  # type: ignore[override]
        spec = self.find_spec(fullname)
        return spec.loader if spec is not None else None

    importlib.machinery.FileFinder.find_module = _file_finder_find_module  # type: ignore[attr-defined]


MODEL_ORDER = [
    "SAM2-Tiny",
    "SAM2.1-Tiny",
    "MobileSAM",
    "MobileSAMV2",
    "YOLO11s-seg",
    "SegEarth-OV",
    "UNet",
    "DeepLabV3+",
]

FIELDS = [
    "Model",
    "Dataset",
    "Perturbation",
    "Strength",
    "mIoU",
    "F1",
    "Accuracy",
    "Delta_mIoU",
    "Delta_F1",
    "Delta_Accuracy",
]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CLIP_MEAN = torch.tensor([122.771, 116.746, 104.094]).view(3, 1, 1)
CLIP_STD = torch.tensor([68.501, 66.632, 70.323]).view(3, 1, 1)


@dataclass(frozen=True)
class Condition:
    perturbation: str
    strength: str
    transform: Callable[[np.ndarray, str], np.ndarray]


def looks_like_project_root(path: Path) -> bool:
    return (path / "1_SAM2_Tiny").is_dir() and (path / "8_DeepLabV3").is_dir()


def discover_project_root() -> Path:
    script_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    candidates = [
        Path(os.environ["VEGETATION_PROJECT_ROOT"]).expanduser()
        if os.environ.get("VEGETATION_PROJECT_ROOT")
        else script_root,
        script_root,
        cwd,
        cwd / "vegetation_models_v2",
        Path("/root/vegetation_models_v2"),
        Path("/root/autodl-tmp/vegetation_models_v2"),
    ]
    for candidate in candidates:
        if looks_like_project_root(candidate):
            return candidate.resolve()
    return script_root


def add_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def remove_path(path: Path) -> None:
    value = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != value]


def purge_module_tree(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            del sys.modules[name]


def load_efficiency_module(root: Path):
    """Reuse the existing checkpoint constructors from the efficiency study."""
    module_path = root / "scripts" / "benchmark_model_efficiency.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Required loader module is missing: {module_path}")
    spec = importlib.util.spec_from_file_location("vegetation_efficiency_loaders", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure_project_root(str(root))
    # MobileSAMV2's fine-tuned semantic head is compatible with the TinyViT
    # MobileSAM encoder used in its original training notebook.  Its source tree
    # is named ``mobilesamv2``, whereas the shared semantic-head loader imports
    # ``mobile_sam``; make that existing source package available explicitly.
    add_path(root / "2_MobileSAM" / "code")
    return module


def pil_from_rgb(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image.astype(np.uint8), mode="RGB")


def brightness(image: np.ndarray, _: str, factor: float) -> np.ndarray:
    return np.asarray(ImageEnhance.Brightness(pil_from_rgb(image)).enhance(factor), dtype=np.uint8)


def contrast(image: np.ndarray, _: str, factor: float) -> np.ndarray:
    return np.asarray(ImageEnhance.Contrast(pil_from_rgb(image)).enhance(factor), dtype=np.uint8)


def gaussian_blur(image: np.ndarray, _: str, kernel: int) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REFLECT_101)


def synthetic_shadow(image: np.ndarray, identifier: str, level: str) -> np.ndarray:
    """Apply a deterministic, smoothly feathered illumination-shadow field.

    The seed is derived from the filename, so every model receives exactly the
    same shadow on a given image.  Mild/moderate/strong increase the covered
    area and reduce the illumination multiplier.
    """
    settings = {
        "mild": (0.82, 0.34, 0.12),
        "moderate": (0.64, 0.52, 0.16),
        "strong": (0.46, 0.68, 0.20),
    }
    minimum, coverage, feather = settings[level]
    height, width = image.shape[:2]
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    angle = (int.from_bytes(digest[:2], "little") / 65535.0 - 0.5) * 0.85
    offset = (int.from_bytes(digest[2:4], "little") / 65535.0 - 0.5) * 0.35
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(width - 1, 1) - 0.5
    y = yy / max(height - 1, 1) - 0.5
    signed_distance = x * np.cos(angle) + y * np.sin(angle) - offset
    transition = max(feather, 1e-4)
    # A logistic transition produces a physically plausible soft shadow edge.
    field = 1.0 / (1.0 + np.exp((signed_distance + coverage / 2.0) / transition))
    multiplier = 1.0 - (1.0 - minimum) * field
    return np.clip(image.astype(np.float32) * multiplier[..., None], 0, 255).astype(np.uint8)


def make_conditions() -> list[Condition]:
    conditions = [Condition("clean", "none", lambda image, identifier: image.copy())]
    conditions.extend(
        Condition("brightness", f"{factor:.2f}", lambda image, identifier, f=factor: brightness(image, identifier, f))
        for factor in (0.70, 0.85, 1.15, 1.30)
    )
    conditions.extend(
        Condition("contrast", f"{factor:.2f}", lambda image, identifier, f=factor: contrast(image, identifier, f))
        for factor in (0.75, 0.90, 1.10, 1.25)
    )
    conditions.extend(
        Condition("gaussian_blur", f"kernel_{kernel}", lambda image, identifier, k=kernel: gaussian_blur(image, identifier, k))
        for kernel in (3, 5, 7)
    )
    conditions.extend(
        Condition("shadow_illumination", level, lambda image, identifier, l=level: synthetic_shadow(image, identifier, l))
        for level in ("mild", "moderate", "strong")
    )
    return conditions


def select_samples(images_dir: Path, masks_dir: Path, max_samples: int) -> list[Path]:
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    images = [path for path in images if (masks_dir / f"{path.stem}.png").is_file()]
    if not images:
        raise RuntimeError(f"No image/mask pairs found under {images_dir} and {masks_dir}")
    if max_samples <= 0 or max_samples >= len(images):
        return images
    # Evenly spaced deterministic selection covers the full ordered acquisition.
    indices = np.linspace(0, len(images) - 1, num=max_samples, dtype=int)
    return [images[index] for index in np.unique(indices)]


def load_ground_truth(mask_path: Path) -> np.ndarray:
    mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
    # labelmap.txt defines background=(0,0,0) and vegetation=(144,32,192).
    return np.any(mask_rgb != 0, axis=2).astype(np.uint8)


def make_tensor(image: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    resized = pil_from_rgb(image).resize((size, size), resample=Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0).to(device, non_blocking=True)


def make_clip_tensor(image: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    resized = pil_from_rgb(image).resize((size, size), resample=Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    tensor = (tensor - CLIP_MEAN) / CLIP_STD
    return tensor.unsqueeze(0).to(device, non_blocking=True)


def semantic_prediction(model: torch.nn.Module, image: np.ndarray, input_size: int, device: torch.device) -> np.ndarray:
    tensor = make_tensor(image, input_size, device)
    with torch.inference_mode():
        logits = model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        prediction = logits.argmax(dim=1, keepdim=True).float()
        prediction = F.interpolate(prediction, size=image.shape[:2], mode="nearest")
    return prediction[0, 0].to("cpu", torch.uint8).numpy()


def segearth_prediction(model: torch.nn.Module, image: np.ndarray, input_size: int, device: torch.device) -> np.ndarray:
    tensor = make_clip_tensor(image, input_size, device)
    with torch.inference_mode():
        prediction = model.predict(tensor.half(), data_samples=None)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        prediction = prediction.float()
        prediction = F.interpolate(prediction.unsqueeze(0), size=image.shape[:2], mode="nearest")
    return prediction[0, 0].to("cpu", torch.uint8).numpy()


def yolo_prediction(yolo, image: np.ndarray, device: torch.device) -> np.ndarray:
    """Convert all predicted instance masks into one binary vegetation mask."""
    result = yolo.predict(
        # Ultralytics' ndarray loader follows OpenCV's BGR convention before
        # converting inputs to RGB internally.  The common perturbation image
        # is RGB, so convert only at this model-specific API boundary.
        source=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        imgsz=512,
        conf=0.001,
        iou=0.7,
        retina_masks=True,
        device=0 if device.type == "cuda" else "cpu",
        verbose=False,
    )[0]
    output = np.zeros(image.shape[:2], dtype=np.uint8)
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return output
    masks = result.masks.data.detach().float().cpu().numpy()
    for mask in masks:
        if mask.shape != output.shape:
            mask = cv2.resize(mask, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST)
        output |= (mask > 0.5).astype(np.uint8)
    return output


def update_confusion(confusion: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    encoded = target.astype(np.int64).ravel() * 2 + prediction.astype(np.int64).ravel()
    confusion += np.bincount(encoded, minlength=4).reshape(2, 2)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    true_negative, false_positive = confusion[0, 0], confusion[0, 1]
    false_negative, true_positive = confusion[1, 0], confusion[1, 1]
    eps = 1e-12
    iou_background = true_negative / (true_negative + false_positive + false_negative + eps)
    iou_vegetation = true_positive / (true_positive + false_positive + false_negative + eps)
    f1 = 2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative + eps)
    accuracy = (true_negative + true_positive) / (confusion.sum() + eps)
    return {"mIoU": (iou_background + iou_vegetation) / 2.0, "F1": f1, "Accuracy": accuracy}


def evaluate_condition(
    predictor: Callable[[np.ndarray], np.ndarray],
    samples: Sequence[Path],
    masks_dir: Path,
    condition: Condition,
) -> dict[str, float]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for image_path in samples:
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        altered = condition.transform(image, image_path.name)
        prediction = predictor(altered)
        target = load_ground_truth(masks_dir / f"{image_path.stem}.png")
        update_confusion(confusion, prediction, target)
    return metrics_from_confusion(confusion)


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def merge_rows(existing_rows: Iterable[dict[str, str]], new_rows: Iterable[dict[str, str]], target_models: set[str]) -> list[dict[str, str]]:
    merged: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in existing_rows:
        if row.get("Model") in target_models:
            continue
        key = (row["Model"], row["Dataset"], row["Perturbation"], row["Strength"])
        merged[key] = {field: row.get(field, "") for field in FIELDS}
    for row in new_rows:
        key = (row["Model"], row["Dataset"], row["Perturbation"], row["Strength"])
        merged[key] = {field: row.get(field, "") for field in FIELDS}
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    condition_order = {
        (condition.perturbation, condition.strength): index
        for index, condition in enumerate(make_conditions())
    }
    return sorted(
        merged.values(),
        key=lambda row: (
            order.get(row["Model"], len(order)),
            row["Dataset"],
            condition_order.get((row["Perturbation"], row["Strength"]), 999),
        ),
    )


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def write_summary(path: Path, rows: list[dict[str, str]], failures: list[str], sample_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parsed = []
    for row in rows:
        parsed.append({**row, **{key: float(row[key]) for key in ("mIoU", "F1", "Accuracy", "Delta_mIoU", "Delta_F1", "Delta_Accuracy")}})
    lines = [
        "# Robustness to Observation-Condition Perturbations",
        "",
        f"- Dataset: held-out `datasets/2024-seg` test subset ({sample_count} fixed images).",
        "- Models: six models with valid fine-tuned checkpoints; no model was retrained.",
        "- Metrics: global two-class mIoU, vegetation F1, and pixel accuracy calculated by this script on the same original-resolution ground-truth masks.",
        "- Deltas: perturbed metric minus the clean metric for the same model; values are proportions, not percentage points.",
        "- Shadow/illumination: deterministic synthetic cast shadows, seeded by filename and therefore identical for every model.",
    ]

    non_clean = [row for row in parsed if row["Perturbation"] != "clean"]
    clean_rows = [row for row in parsed if row["Perturbation"] == "clean"]
    model_deltas: dict[str, list[float]] = {}
    perturbation_deltas: dict[str, list[float]] = {}
    for row in non_clean:
        model_deltas.setdefault(row["Model"], []).append(row["Delta_mIoU"])
        perturbation_deltas.setdefault(row["Perturbation"], []).append(row["Delta_mIoU"])

    if clean_rows and non_clean:
        lines.extend(
            [
                "",
                "## Clean baseline and robustness summary",
                "",
                "| Model | Clean mIoU | Clean F1 | Clean Accuracy | Mean Delta_mIoU | Worst Delta_mIoU | Worst perturbation |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for clean in sorted(clean_rows, key=lambda item: item["mIoU"], reverse=True):
            model_rows = [row for row in non_clean if row["Model"] == clean["Model"]]
            worst = min(model_rows, key=lambda item: item["Delta_mIoU"])
            mean_delta = np.mean([row["Delta_mIoU"] for row in model_rows])
            worst_name = f"{worst['Perturbation']}, {worst['Strength']}"
            lines.append(
                f"| {clean['Model']} | {clean['mIoU']:.4f} | {clean['F1']:.4f} | {clean['Accuracy']:.4f} | "
                f"{mean_delta:+.4f} | {worst['Delta_mIoU']:+.4f} | {worst_name} |"
            )

        lines.extend(
            [
                "",
                "## Perturbation-category summary",
                "",
                "| Perturbation category | Mean Delta_mIoU |",
                "|---|---:|",
            ]
        )
        for name, values in sorted(perturbation_deltas.items(), key=lambda item: np.mean(item[1])):
            lines.append(f"| {name} | {np.mean(values):+.4f} |")

        best_model, best_values = max(model_deltas.items(), key=lambda item: np.mean(item[1]))
        worst_perturbation, worst_values = min(perturbation_deltas.items(), key=lambda item: np.mean(item[1]))
        lines.extend(
            [
                "",
                "## Paper-table suggestion",
                "",
                "Use the clean-baseline robustness table as the compact manuscript table. If space allows, add the perturbation-category summary as a second short table.",
                "",
                "Suggested paragraph:",
                "",
                f"> To examine adaptation to observation-condition changes, we conducted an inference-only robustness test using the same fixed {sample_count}-image held-out subset from the 2024 segmentation dataset. Brightness, contrast, Gaussian blur, and synthetic shadow/illumination perturbations were applied identically to all completed models, while ground-truth masks were unchanged. Among the completed models, {best_model} showed the smallest average mIoU degradation under perturbations (mean Delta_mIoU = {np.mean(best_values):+.4f}). The most challenging condition overall was {worst_perturbation} (mean Delta_mIoU = {np.mean(worst_values):+.4f}).",
                "",
                "## Revision response draft",
                "",
                f"> Response to Reviewer 2, Comment 4: Thank you for the helpful suggestion. We added a lightweight inference-only robustness analysis to evaluate adaptation to common observation-condition variations. Using the same fixed held-out subset, we applied brightness, contrast, Gaussian blur, and synthetic shadow/illumination perturbations to the input images and evaluated all completed models with the same pixel-level mIoU, F1, and accuracy implementation. The results show that {best_model} is comparatively stable across perturbations, whereas {worst_perturbation} is the most damaging observation change on average. We report these results in the robustness summary table and provide the experimental scripts and CSV outputs for reproducibility.",
            ]
        )

    lines.extend(
        [
            "",
            "## Per-condition results",
            "",
            "| Model | Perturbation | Strength | mIoU | F1 | Accuracy | Delta_mIoU | Delta_F1 | Delta_Accuracy |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in parsed:
        lines.append(
            "| {Model} | {Perturbation} | {Strength} | {mIoU:.4f} | {F1:.4f} | {Accuracy:.4f} | "
            "{Delta_mIoU:+.4f} | {Delta_F1:+.4f} | {Delta_Accuracy:+.4f} |".format(**row)
        )

    if non_clean:
        lines.extend(
            [
                "",
                "## Robustness ranking",
                "",
                "Higher mean Delta_mIoU (i.e., a smaller average loss from the clean condition) indicates stronger robustness.",
                "",
                "| Rank | Model | Mean Delta_mIoU over all 14 perturbation settings |",
                "|---:|---|---:|",
            ]
        )
        for rank, (model, values) in enumerate(sorted(model_deltas.items(), key=lambda item: np.mean(item[1]), reverse=True), start=1):
            lines.append(f"| {rank} | {model} | {np.mean(values):+.4f} |")
        lines.extend(
            [
                "",
                "## Perturbation sensitivity",
                "",
                "More negative mean Delta_mIoU indicates a more damaging perturbation category, averaged over tested strengths and completed models.",
                "",
                "| Perturbation | Mean Delta_mIoU |",
                "|---|---:|",
            ]
        )
        for name, values in sorted(perturbation_deltas.items(), key=lambda item: np.mean(item[1])):
            lines.append(f"| {name} | {np.mean(values):+.4f} |")

    if failures:
        lines.extend(["", "## Models not completed", ""])
        lines.extend(f"- {failure}" for failure in failures)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_predictor(model_name: str, root: Path, device: torch.device, loaders):
    if model_name == "YOLO11s-seg":
        # MobileSAMV2 vendors an older package named ``ultralytics``.  If that
        # path stays ahead of site-packages, YOLO11 checkpoints are unpickled
        # with the wrong module definitions.
        remove_path(root / "6_MobileSAMV2" / "code")
        purge_module_tree("ultralytics")
        from ultralytics import YOLO

        checkpoint = root / "3_YOLO11seg" / "checkpoints" / "yolo11s_vegetation" / "weights" / "best.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f"Invalid YOLO checkpoint: {checkpoint}")
        yolo = YOLO(str(checkpoint))
        return yolo, lambda image: yolo_prediction(yolo, image, device)

    if model_name == "SAM2-Tiny":
        # Importing ``sam2`` initializes its own Hydra config module.  Do not
        # wrap this in initialize_config_dir (as an older benchmark wrapper
        # did), because that is a second initialization and can raise
        # GlobalHydra-already-initialized on long sequential runs.
        from hydra.core.global_hydra import GlobalHydra

        global_hydra = GlobalHydra.instance()
        if global_hydra.is_initialized():
            global_hydra.clear()
        add_path(root / "1_SAM2_Tiny" / "code")
        purge_module_tree("sam2")
        from sam2.build_sam import build_sam2

        backbone = build_sam2(
            "configs/sam2/sam2_hiera_t.yaml",
            ckpt_path=str(root / "1_SAM2_Tiny" / "weights" / "sam2_hiera_tiny.pt"),
            device=device,
            mode="eval",
        )
        model = loaders.EncoderSegmentationModel(backbone.image_encoder)
        checkpoint = root / "1_SAM2_Tiny" / "checkpoints" / "sam2tiny_best.pth"
        model.load_state_dict(loaders.get_state_dict(checkpoint))
        model = model.to(device).eval()
        input_size = 1024
    elif model_name == "SAM2.1-Tiny":
        checkpoint = root / "5_SAM21_Tiny" / "checkpoints" / "sam21tiny_best.pth"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f"Invalid SAM2.1 fine-tuned checkpoint: {checkpoint}")
        model, input_size, _notes = loaders.LOADERS[model_name](device)
    elif model_name == "SegEarth-OV":
        local_clip = root / "4_SegEarth_OV" / "weights" / "pytorch_model.bin"
        upsampler = root / "4_SegEarth_OV" / "code" / "simfeatup_dev" / "weights" / "xclip_jbu_one_million_aid.ckpt"
        if not local_clip.is_file() or local_clip.stat().st_size == 0:
            raise FileNotFoundError(f"Invalid SegEarth CLIP checkpoint: {local_clip}")
        if not upsampler.is_file() or upsampler.stat().st_size == 0:
            raise FileNotFoundError(f"Invalid SegEarth SimFeatUp checkpoint: {upsampler}")
        model, input_size, _notes = loaders.LOADERS[model_name](device)
        wrapped = getattr(model, "wrapped", model)
        return wrapped, lambda image: segearth_prediction(wrapped, image, input_size, device)
    else:
        model, input_size, _notes = loaders.LOADERS[model_name](device)
        if model_name == "UNet":
            # The UNet training/evaluation notebook uses 512x512 inputs.
            # Override the efficiency benchmark's throughput-oriented size.
            input_size = 512
    return model, lambda image: semantic_prediction(model, image, input_size, device)


def main() -> None:
    default_root = discover_project_root()
    parser = argparse.ArgumentParser(description="Evaluate robustness to brightness, contrast, blur, and illumination perturbations.")
    parser.add_argument("--project-root", default=str(default_root), help="Project root (e.g. /root/vegetation_models_v2).")
    parser.add_argument("--models", nargs="*", choices=MODEL_ORDER, default=MODEL_ORDER)
    parser.add_argument("--max-samples", type=int, default=32, help="Fixed representative test subset size; use 0 for all 67 test images.")
    parser.add_argument("--smoke", action="store_true", help="Fast diagnostic: first 4 samples and clean/one setting per perturbation only.")
    parser.add_argument("--output-csv", default=None, help="Override results/robustness_perturbation_results.csv.")
    parser.add_argument("--output-summary", default=None, help="Override results/robustness_perturbation_summary.md.")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    if not looks_like_project_root(root):
        raise SystemExit(f"{root} is not the project root (expected 1_SAM2_Tiny through 8_DeepLabV3).")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the requested checkpoint evaluation.")
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be 0 (full test set) or a positive integer.")

    results_dir = root / "results"
    csv_path = Path(args.output_csv).expanduser().resolve() if args.output_csv else results_dir / "robustness_perturbation_results.csv"
    summary_path = Path(args.output_summary).expanduser().resolve() if args.output_summary else results_dir / "robustness_perturbation_summary.md"
    images_dir = root / "datasets" / "2024-seg" / "JPEGImages"
    masks_dir = root / "datasets" / "2024-seg" / "SegmentationClass"
    samples = select_samples(images_dir, masks_dir, args.max_samples)
    if args.smoke:
        samples = samples[: min(4, len(samples))]
    conditions = make_conditions()
    if args.smoke:
        keep = {("clean", "none"), ("brightness", "0.70"), ("contrast", "0.75"), ("gaussian_blur", "kernel_7"), ("shadow_illumination", "strong")}
        conditions = [condition for condition in conditions if (condition.perturbation, condition.strength) in keep]

    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = results_dir / "robustness_perturbation_subset_manifest.txt"
    manifest.write_text(
        "# Fixed held-out 2024 test subset used by run_robustness_perturbation_eval.py\n"
        f"# Sample count: {len(samples)}\n"
        f"# Mode: {'smoke' if args.smoke else 'formal'}\n"
        + "\n".join(path.name for path in samples)
        + "\n",
        encoding="utf-8",
    )

    print(f"Repository: {root}")
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}")
    print(f"Dataset: {images_dir} / {masks_dir}")
    print(f"Samples: {len(samples)} | Conditions: {len(conditions)} | Models: {', '.join(args.models)}")
    print(f"Subset manifest: {manifest}")
    loaders = load_efficiency_module(root)
    device = torch.device("cuda:0")
    existing_rows = read_csv(csv_path)
    target_models = set(args.models)
    completed_rows: list[dict[str, str]] = []
    failures: list[str] = []

    for model_name in args.models:
        print(f"\n[{model_name}] loading existing checkpoint...")
        model = None
        try:
            model, predictor = build_predictor(model_name, root, device, loaders)
            metric_rows: dict[tuple[str, str], dict[str, float]] = {}
            for index, condition in enumerate(conditions, start=1):
                print(f"[{model_name}] {index}/{len(conditions)} {condition.perturbation}:{condition.strength}")
                metric_rows[(condition.perturbation, condition.strength)] = evaluate_condition(predictor, samples, masks_dir, condition)
            clean = metric_rows[("clean", "none")]
            for condition in conditions:
                metric = metric_rows[(condition.perturbation, condition.strength)]
                completed_rows.append(
                    {
                        "Model": model_name,
                        "Dataset": f"2024-seg fixed subset (n={len(samples)})",
                        "Perturbation": condition.perturbation,
                        "Strength": condition.strength,
                        "mIoU": format_metric(metric["mIoU"]),
                        "F1": format_metric(metric["F1"]),
                        "Accuracy": format_metric(metric["Accuracy"]),
                        "Delta_mIoU": format_metric(metric["mIoU"] - clean["mIoU"]),
                        "Delta_F1": format_metric(metric["F1"] - clean["F1"]),
                        "Delta_Accuracy": format_metric(metric["Accuracy"] - clean["Accuracy"]),
                    }
                )
            merged_rows = merge_rows(existing_rows, completed_rows, target_models)
            write_csv(csv_path, merged_rows)
            write_summary(summary_path, merged_rows, failures, len(samples))
        except Exception as exc:  # Preserve other completed models for long GPU runs.
            short_error = " ".join(traceback.format_exception_only(type(exc), exc)).strip().replace("\n", " ")
            failures.append(f"`{model_name}` was not completed: {short_error[:800]}")
            print(f"[{model_name}] FAILED: {short_error}")
            merged_rows = merge_rows(existing_rows, completed_rows, target_models)
            write_csv(csv_path, merged_rows)
            write_summary(summary_path, merged_rows, failures, len(samples))
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    merged_rows = merge_rows(existing_rows, completed_rows, target_models)
    write_csv(csv_path, merged_rows)
    write_summary(summary_path, merged_rows, failures, len(samples))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")
    if failures:
        raise SystemExit(f"Completed {len(completed_rows)} rows with {len(failures)} model failure(s); see {summary_path}.")


if __name__ == "__main__":
    main()
