#!/usr/bin/env python3
"""Prompt-based Qwen2.5-VL vegetation localization supplement.

This is not dense semantic-segmentation training. It asks Qwen2.5-VL-3B to
return vegetation bounding boxes, converts boxes to coarse binary masks, and
reports supplemental box-mask metrics against the binary vegetation masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DATASETS = ("zijinshan", "loveda", "potsdam", "vaihingen")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass(frozen=True)
class Sample:
    dataset: str
    split: str
    name: str
    image: Path
    mask: Path
    round_index: int


@dataclass(frozen=True)
class SampleCandidate:
    name: str
    image: Path
    mask: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL-3B prompt-based vegetation localization.")
    parser.add_argument("--project-root", type=Path, default=Path("/content/vegetation_models_v2"))
    parser.add_argument("--binary-root", type=Path, default=Path("/content/binary"))
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--split", choices=["auto", "default", "test", "val"], default="auto")
    parser.add_argument("--max-samples-per-dataset", type=int, default=16)
    parser.add_argument(
        "--rounds-per-dataset",
        type=int,
        default=None,
        help="Alias for --max-samples-per-dataset; useful for prompt-eval runs described as 100 rounds.",
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument(
        "--coordinate-mode",
        choices=["auto", "normalized1000", "pixel"],
        default="auto",
        help="How to interpret returned boxes. Auto treats boxes within image dimensions as pixel coordinates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate split resolution and sample counts; do not load Qwen or run inference.",
    )
    return parser.parse_args()


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
        if stems:
            return requested, stems
        raise FileNotFoundError(f"Missing split {requested}.txt under {dataset_root}")
    for split_name in ("test", "val", "default"):
        stems = read_split(dataset_root, split_name)
        if stems:
            return split_name, stems
    raise FileNotFoundError(f"No usable split file under {dataset_root}")


def valid_candidates(dataset_root: Path, stems: list[str]) -> tuple[list[SampleCandidate], list[str]]:
    candidates: list[SampleCandidate] = []
    missing: list[str] = []
    for stem in stems:
        image = existing_file(dataset_root / "JPEGImages", stem)
        mask = existing_file(dataset_root / "SegmentationClass", stem)
        if image and mask:
            candidates.append(SampleCandidate(stem, image, mask))
        else:
            missing.append(stem)
    return candidates, missing


def collect_samples(
    binary_root: Path,
    datasets: list[str],
    split: str,
    max_samples: int,
    seed: int,
    exact_rounds: bool = False,
) -> list[Sample]:
    rng = random.Random(seed)
    samples: list[Sample] = []
    for dataset in datasets:
        root = binary_root / dataset
        split_name, stems = resolve_split(root, split)
        candidates, missing = valid_candidates(root, stems)
        if not candidates:
            raise FileNotFoundError(
                f"No valid image-mask pairs for dataset={dataset}, split={split_name} under {root}"
            )
        if exact_rounds:
            if len(candidates) >= max_samples:
                selected = candidates[:]
                rng.shuffle(selected)
                selected = selected[:max_samples]
            else:
                selected = [rng.choice(candidates) for _ in range(max_samples)]
        else:
            selected = candidates[:]
            rng.shuffle(selected)
            selected = selected[:max_samples]
        print(
            f"{dataset}: split={split_name}, split_stems={len(stems)}, "
            f"valid_pairs={len(candidates)}, missing_pairs={len(missing)}, "
            f"scheduled_rounds={len(selected)}"
        )
        for round_index, candidate in enumerate(selected, start=1):
            samples.append(
                Sample(dataset, split_name, candidate.name, candidate.image, candidate.mask, round_index)
            )
    return samples


def load_qwen(args: argparse.Namespace):
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Missing Qwen-VL dependencies. In Colab run: "
            "pip install -q -U transformers accelerate qwen-vl-utils"
        ) from exc
    dtype = "auto"
    if args.torch_dtype == "float16":
        dtype = torch.float16
    elif args.torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.torch_dtype == "float32":
        dtype = torch.float32
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            device_map="auto" if args.device == "cuda" else None,
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
    except ImportError as exc:
        raise RuntimeError(
            "Missing Qwen2.5-VL optional dependencies. In Colab run:\n"
            "pip install -q -U transformers accelerate qwen-vl-utils\n"
            "Then rerun this script."
        ) from exc
    return model, processor, process_vision_info


def prompt_for_image(image_path: Path) -> list[dict[str, object]]:
    prompt = (
        "You are evaluating remote-sensing vegetation localization. "
        "Find all visible vegetation or forest regions. Return only valid JSON with this schema: "
        "{\"vegetation_present\": true/false, \"boxes\": [[x1,y1,x2,y2], ...], \"confidence\": 0.0}. "
        "Coordinates must be normalized integers from 0 to 1000 relative to image width and height. "
        "Use a small number of boxes that cover vegetation regions; do not include explanation text."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def run_one(model, processor, process_vision_info, image_path: Path, max_new_tokens: int) -> str:
    messages = prompt_for_image(image_path)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def parse_json_response(text: str) -> dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {"vegetation_present": False, "boxes": [], "confidence": 0.0, "parse_error": text[:300]}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"vegetation_present": False, "boxes": [], "confidence": 0.0, "parse_error": text[:300]}
    boxes = data.get("boxes", [])
    if not isinstance(boxes, list):
        boxes = []
    clean_boxes = []
    for box in boxes:
        if isinstance(box, list) and len(box) == 4:
            try:
                clean_boxes.append([float(v) for v in box])
            except Exception:
                pass
    data["boxes"] = clean_boxes
    return data


def box_uses_pixel_coordinates(box: list[float], size: tuple[int, int]) -> bool:
    width, height = size
    x1, y1, x2, y2 = box
    return max(x1, x2) <= width * 1.05 and max(y1, y2) <= height * 1.05


def boxes_to_mask(
    boxes: list[list[float]],
    size: tuple[int, int],
    coordinate_mode: str = "auto",
) -> tuple[np.ndarray, dict[str, int]]:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    modes = {"pixel": 0, "normalized1000": 0}
    for x1, y1, x2, y2 in boxes:
        use_pixel = coordinate_mode == "pixel" or (
            coordinate_mode == "auto" and box_uses_pixel_coordinates([x1, y1, x2, y2], size)
        )
        if use_pixel:
            modes["pixel"] += 1
            x1i = int(max(0, min(width, round(x1))))
            x2i = int(max(0, min(width, round(x2))))
            y1i = int(max(0, min(height, round(y1))))
            y2i = int(max(0, min(height, round(y2))))
        else:
            modes["normalized1000"] += 1
            x1i = int(max(0, min(width, round(x1 / 1000.0 * width))))
            x2i = int(max(0, min(width, round(x2 / 1000.0 * width))))
            y1i = int(max(0, min(height, round(y1 / 1000.0 * height))))
            y2i = int(max(0, min(height, round(y2 / 1000.0 * height))))
        if x2i > x1i and y2i > y1i:
            mask[y1i:y2i, x1i:x2i] = 1
    return mask, modes


def mask_metrics(pred: np.ndarray, target_path: Path, size: tuple[int, int]) -> dict[str, float]:
    target = Image.open(target_path).convert("RGB").resize(size, Image.Resampling.NEAREST)
    target_np = (np.any(np.asarray(target, dtype=np.uint8) != 0, axis=2)).astype(np.uint8)
    tp = int(((pred == 1) & (target_np == 1)).sum())
    fp = int(((pred == 1) & (target_np == 0)).sum())
    fn = int(((pred == 0) & (target_np == 1)).sum())
    tn = int(((pred == 0) & (target_np == 0)).sum())
    eps = 1e-12
    return {
        "box_mask_iou": tp / (tp + fp + fn + eps),
        "box_mask_f1": 2 * tp / (2 * tp + fp + fn + eps),
        "box_mask_accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = args.project_root / "11_QwenVL" / "prompt_eval"
    if args.results_csv is None:
        args.results_csv = args.project_root / "results" / "qwenvl_prompt_eval_results.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    exact_rounds = args.rounds_per_dataset is not None
    max_samples = args.rounds_per_dataset or args.max_samples_per_dataset
    samples = collect_samples(args.binary_root, args.datasets, args.split, max_samples, args.seed, exact_rounds)
    if exact_rounds:
        expected = len(args.datasets) * max_samples
        if len(samples) != expected:
            raise RuntimeError(
                f"Expected exactly {expected} scheduled rounds "
                f"({len(args.datasets)} datasets x {max_samples}), got {len(samples)}."
            )
    print(f"Total scheduled QwenVL prompt-eval rounds: {len(samples)}")
    if args.dry_run:
        return 0
    model, processor, process_vision_info = load_qwen(args)
    rows = []
    jsonl_path = args.output_dir / "qwenvl_raw_outputs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for idx, sample in enumerate(samples, start=1):
            image = Image.open(sample.image).convert("RGB")
            raw = run_one(model, processor, process_vision_info, sample.image, args.max_new_tokens)
            parsed = parse_json_response(raw)
            boxes = parsed.get("boxes", [])
            pred_mask, coordinate_modes = boxes_to_mask(
                boxes if isinstance(boxes, list) else [],
                image.size,
                args.coordinate_mode,
            )
            metrics = mask_metrics(pred_mask, sample.mask, image.size)
            row = {
                "dataset": sample.dataset,
                "split": sample.split,
                "round_index": sample.round_index,
                "sample": sample.name,
                "model": args.model_id,
                "box_count": len(boxes) if isinstance(boxes, list) else 0,
                "coordinate_mode": args.coordinate_mode,
                "pixel_box_count": coordinate_modes["pixel"],
                "normalized1000_box_count": coordinate_modes["normalized1000"],
                "vegetation_present": parsed.get("vegetation_present"),
                "confidence": parsed.get("confidence"),
                **metrics,
            }
            rows.append(row)
            jsonl.write(json.dumps({"row": row, "parsed": parsed, "raw": raw}, ensure_ascii=False) + "\n")
            print(
                f"[{idx}/{len(samples)}] {sample.dataset}/round_{sample.round_index:03d}/{sample.name}: "
                f"boxes={row['box_count']}, IoU={row['box_mask_iou']:.4f}"
            )
    with args.results_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {args.results_csv}")
    print(f"Wrote raw JSONL: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
