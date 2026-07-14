#!/usr/bin/env python3
"""LocateAnything-only Colab efficiency and perturbation runner.

Run this script with the isolated LocateAnything package path, e.g.

    PYTHONPATH=/content/la_pkgs:$PYTHONPATH python scripts/colab_locateanything_efficiency_perturbation.py --mode both

The script intentionally stays separate from the DINO/Qwen runner because
LocateAnything needs pinned dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance


@dataclass(frozen=True)
class Condition:
    perturbation: str
    strength: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LocateAnything efficiency and perturbation runner.")
    parser.add_argument("--project-root", type=Path, default=Path("/content/vegetation_models_v2"))
    parser.add_argument("--binary-root", type=Path, default=Path("/content/binary"))
    parser.add_argument("--model-id", default="nvidia/LocateAnything-3B")
    parser.add_argument("--datasets", nargs="+", default=["zijinshan"])
    parser.add_argument("--split", choices=["auto", "test", "val", "default"], default="auto")
    parser.add_argument("--rounds-per-dataset", type=int, default=67)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["efficiency", "perturbation", "both"], default="both")
    parser.add_argument("--efficiency-csv", type=Path, default=None)
    parser.add_argument("--robustness-csv", type=Path, default=None)
    parser.add_argument("--detail-csv", type=Path, default=None)
    parser.add_argument("--perturbed-image-root", type=Path, default=None)
    return parser.parse_args()


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def perturb_image(img: Image.Image, condition: Condition, identifier: str) -> Image.Image:
    img = img.convert("RGB")
    if condition.perturbation == "clean":
        return img
    if condition.perturbation == "brightness":
        return ImageEnhance.Brightness(img).enhance(float(condition.strength))
    if condition.perturbation == "contrast":
        return ImageEnhance.Contrast(img).enhance(float(condition.strength))
    if condition.perturbation == "gaussian_blur":
        import cv2

        kernel = int(condition.strength.replace("kernel_", ""))
        arr = np.asarray(img, dtype=np.uint8)
        out = cv2.GaussianBlur(arr, (kernel, kernel), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REFLECT_101)
        return Image.fromarray(out.astype(np.uint8), mode="RGB")
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
        field = 1.0 / (1.0 + np.exp((signed_distance + coverage / 2.0) / max(feather, 1e-4)))
        multiplier = 1.0 - (1.0 - minimum) * field
        out = np.clip(image.astype(np.float32) * multiplier[..., None], 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")
    raise ValueError(condition)


def save_perturbed(sample, condition: Condition, root: Path) -> Path:
    out_dir = root / sample.dataset / condition.perturbation / condition.strength
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{sample.name}.png"
    if not out.exists():
        image = Image.open(sample.image).convert("RGB")
        perturb_image(image, condition, sample.image.name).save(out)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        print(f"No rows for {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def input_size_label(samples) -> str:
    sizes = []
    for sample in samples:
        with Image.open(sample.image) as image:
            sizes.append(image.size)
    unique = sorted(set(sizes))
    if len(unique) == 1:
        width, height = unique[0]
        return f"{width}x{height} original"
    return "mixed original sizes: " + ", ".join(f"{w}x{h}" for w, h in unique[:5])


def locateanything_prompt() -> str:
    return "Locate all the instances that match the following description: vegetation</c>tree</c>forest</c>grass</c>crop</c>shrub."


def build_inputs(processor, image_path: Path, device: str):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": locateanything_prompt()}]}]
    if hasattr(processor, "py_apply_chat_template"):
        text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if hasattr(processor, "process_vision_info"):
        images, videos = processor.process_vision_info(messages)
    else:
        images, videos = [image], None
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16 if device == "cuda" else torch.float32)
    return inputs


def estimate_flops_g(model, processor, sample_image: Path, device: str, params_m: float) -> tuple[object, str]:
    """Estimate LocateAnything prompt prefill FLOPs for a representative image.

    LocateAnything is generation-based, so the complete cost depends on the
    generated sequence length. This reports a reproducible prompt prefill
    estimate for the same image/prompt path used by the grounding run.
    """
    try:
        inputs = build_inputs(processor, sample_image, device)
        flops = None
        if hasattr(model, "floating_point_ops"):
            try:
                flops = model.floating_point_ops(inputs)
            except Exception:
                flops = None
        if flops is None:
            token_count = int(inputs["input_ids"].numel()) if "input_ids" in inputs else 0
            patch_count = int(inputs["pixel_values"].shape[0]) if "pixel_values" in inputs else 0
            effective_tokens = max(1, token_count + patch_count)
            flops = 2.0 * (params_m * 1e6) * effective_tokens
            note = (
                "Estimated prompt prefill FLOPs as 2 x parameter_count x "
                "(text tokens + visual tokens/patches); generation-loop FLOPs are output-length dependent."
            )
        else:
            note = (
                "Estimated prompt prefill FLOPs using model.floating_point_ops on one representative image; "
                "generation-loop FLOPs are output-length dependent."
            )
        return round(float(flops) / 1e9, 4), note
    except Exception as exc:
        return "pending_generation_dependent", (
            "FLOPs estimation failed in this environment; LocateAnything grounding is generation-dependent. "
            f"Error: {type(exc).__name__}: {exc}"
        )


def run_one_metric(la, model, processor, tokenizer, image_path: Path, mask_path: Path, args: argparse.Namespace) -> tuple[dict[str, float], int, float]:
    image = Image.open(image_path).convert("RGB")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    raw = la.run_one(model, processor, tokenizer, image_path, args.max_new_tokens, args.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    parsed = la.parse_json_response(raw)
    boxes = parsed.get("boxes", [])
    pred_mask = la.boxes_to_mask(boxes if isinstance(boxes, list) else [], image.size)
    metrics = la.mask_metrics(pred_mask, mask_path, image.size)
    return metrics, len(boxes) if isinstance(boxes, list) else 0, latency_ms


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["dataset"]), str(row["perturbation"]), str(row["strength"]))
        grouped.setdefault(key, []).append(row)
    first_pass = {}
    summary = []
    for (model, dataset, perturbation, strength), group in grouped.items():
        iou = float(np.mean([float(r["box_mask_iou"]) for r in group]))
        f1 = float(np.mean([float(r["box_mask_f1"]) for r in group]))
        acc = float(np.mean([float(r["box_mask_accuracy"]) for r in group]))
        boxes = float(np.mean([float(r["box_count"]) for r in group]))
        if perturbation == "clean" and strength == "none":
            first_pass[(model, dataset)] = (iou, f1, acc)
        summary.append((model, dataset, perturbation, strength, len(group), iou, f1, acc, boxes))

    out = []
    for model, dataset, perturbation, strength, n, iou, f1, acc, boxes in summary:
        clean_iou, clean_f1, clean_acc = first_pass.get((model, dataset), (iou, f1, acc))
        out.append(
            {
                "model": model,
                "dataset": dataset,
                "task_type": "grounding",
                "metric_type": "box-mask",
                "perturbation": perturbation,
                "strength": strength,
                "n": n,
                "mIoU": iou,
                "F1": f1,
                "Accuracy": acc,
                "Delta_mIoU": iou - clean_iou,
                "Delta_F1": f1 - clean_f1,
                "Delta_Accuracy": acc - clean_acc,
                "box_mask_iou": iou,
                "box_mask_f1": f1,
                "box_mask_accuracy": acc,
                "avg_boxes": boxes,
            }
        )
    return out


def main() -> int:
    args = parse_args()
    results_dir = args.project_root / "results"
    args.efficiency_csv = args.efficiency_csv or (results_dir / "four_model_locateanything_efficiency_colab.csv")
    args.robustness_csv = args.robustness_csv or (results_dir / "four_model_locateanything_robustness_colab.csv")
    args.detail_csv = args.detail_csv or (results_dir / "four_model_locateanything_detail_colab.csv")
    args.perturbed_image_root = args.perturbed_image_root or (results_dir / "locateanything_perturbed_images")

    la = import_file(args.project_root / "scripts/run_locateanything_vegetation_grounding.py", "locateanything_base")
    samples = la.collect_samples(args.binary_root, args.datasets, args.split, args.rounds_per_dataset, args.seed, exact_rounds=False)
    model, processor, tokenizer = la.load_model(args.model_id, args.device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device

    if args.mode in {"efficiency", "both"}:
        clean_rows = []
        flops_g, flops_note = estimate_flops_g(model, processor, samples[0].image, args.device, params_m)
        for sample in samples:
            metrics, box_count, latency_ms = run_one_metric(la, model, processor, tokenizer, sample.image, sample.mask, args)
            clean_rows.append({"latency_ms": latency_ms, "box_count": box_count, **metrics})
            print(f"[LA efficiency] {sample.dataset}/{sample.name}: {latency_ms:.1f} ms, IoU={metrics['box_mask_iou']:.4f}")
        mean_ms = float(np.mean([r["latency_ms"] for r in clean_rows]))
        write_csv(
            args.efficiency_csv,
            [
                {
                    "model": args.model_id,
                    "dataset_head": ",".join(args.datasets),
                    "task_type": "grounding",
                    "params_M": params_m,
                    "trainable_params_M": "N/A",
                    "flops_G": flops_g,
                    "flops_note": flops_note,
                    "inference_time_ms": mean_ms,
                    "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
                    "gpu_memory_MB": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0,
                    "input_size": input_size_label(samples),
                    "device": device_name,
                    "notes": f"Prompt-to-box grounding latency, n={len(samples)}, max_new_tokens={args.max_new_tokens}; isolated PYTHONPATH environment required.",
                }
            ],
        )

    if args.mode in {"perturbation", "both"}:
        detail_rows = []
        for condition in conditions():
            for sample in samples:
                image_path = sample.image if condition.perturbation == "clean" else save_perturbed(sample, condition, args.perturbed_image_root)
                metrics, box_count, latency_ms = run_one_metric(la, model, processor, tokenizer, image_path, sample.mask, args)
                row = {
                    "model": args.model_id,
                    "dataset": sample.dataset,
                    "sample": sample.name,
                    "round_index": sample.round_index,
                    "perturbation": condition.perturbation,
                    "strength": condition.strength,
                    "box_count": box_count,
                    "latency_ms": latency_ms,
                    **metrics,
                }
                detail_rows.append(row)
                print(f"[LA perturb] {condition.perturbation}:{condition.strength} {sample.dataset}/{sample.name}: IoU={metrics['box_mask_iou']:.4f}")
        write_csv(args.detail_csv, detail_rows)
        write_csv(args.robustness_csv, summarize(detail_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
