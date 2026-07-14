#!/usr/bin/env python3
"""Prompt-based LocateAnything vegetation grounding supplement.

This script treats LocateAnything as a grounding model, not as a dense
segmentation trainer. It asks for vegetation boxes, converts them to coarse
binary masks, and writes supplemental box-mask metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LocateAnything prompt-based vegetation grounding.")
    parser.add_argument("--project-root", type=Path, default=Path("/content/vegetation_models_v2"))
    parser.add_argument("--binary-root", type=Path, default=Path("/content/binary"))
    parser.add_argument("--model-id", default="nvidia/LocateAnything-3B")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--split", choices=["auto", "default", "test", "val"], default="auto")
    parser.add_argument("--max-samples-per-dataset", type=int, default=16)
    parser.add_argument(
        "--rounds-per-dataset",
        type=int,
        default=None,
        help="Alias for --max-samples-per-dataset; useful for grounding runs described as 100 rounds.",
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
        valid_stems = [
            stem
            for stem in stems
            if existing_file(root / "JPEGImages", stem) and existing_file(root / "SegmentationClass", stem)
        ]
        if not valid_stems:
            raise FileNotFoundError(f"No valid image/mask pairs found for {dataset} under {root}")
        rng.shuffle(valid_stems)
        if exact_rounds:
            selected_stems = [rng.choice(valid_stems) for _ in range(max_samples)]
        else:
            selected_stems = valid_stems[:max_samples]
        for round_index, stem in enumerate(selected_stems, start=1):
            image = existing_file(root / "JPEGImages", stem)
            mask = existing_file(root / "SegmentationClass", stem)
            assert image is not None and mask is not None
            samples.append(Sample(dataset, split_name, stem, image, mask, round_index))
    return samples


def load_model(model_id: str, device: str):
    try:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Missing dependencies. In Colab run: pip install -q -U transformers accelerate") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device).eval()
    except ImportError as exc:
        raise RuntimeError(
            "LocateAnything remote code is missing optional packages. In Colab run:\n"
            "pip install -q opencv-python-headless==4.11.0.86 transformers==4.57.1 peft torchvision decord==0.6.0 lmdb==1.7.5 accelerate\n"
            "Then rerun this script."
        ) from exc
    except AttributeError as exc:
        raise RuntimeError(
            "LocateAnything remote code is incompatible with the current transformers version. In Colab run:\n"
            "pip install -q --force-reinstall transformers==4.57.1 decord==0.6.0 lmdb==1.7.5 peft accelerate\n"
            "Then rerun this script."
        ) from exc
    return model, processor, tokenizer


def prompt() -> str:
    return "Locate all the instances that match the following description: vegetation</c>tree</c>forest</c>grass</c>crop</c>shrub."


def run_one(model, processor, tokenizer, image_path: Path, max_new_tokens: int, device: str) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt()}]}]
    if hasattr(processor, "py_apply_chat_template"):
        text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if hasattr(processor, "process_vision_info"):
        images, videos = processor.process_vision_info(messages)
    else:
        images, videos = [image], None
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"].to(torch.float16 if device == "cuda" else torch.float32)
    image_grid_hws = inputs.get("image_grid_hws", None)
    with torch.no_grad():
        response = model.generate(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode="hybrid",
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )
    response_text = response[0] if isinstance(response, tuple) else response
    return str(response_text)


def parse_json_response(text: str) -> dict[str, object]:
    structured_boxes = []
    for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", text):
        structured_boxes.append([float(value) for value in match.groups()])
    if structured_boxes:
        return {"boxes": structured_boxes, "confidence": None, "raw_format": "locateanything_box_tokens"}
    if re.search(r"<box>\s*None\s*</box>", text, flags=re.I):
        return {"boxes": [], "confidence": None, "raw_format": "locateanything_none_box_tokens"}
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {"boxes": [], "confidence": 0.0, "parse_error": text[:300]}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"boxes": [], "confidence": 0.0, "parse_error": text[:300]}
    boxes = data.get("boxes", [])
    clean_boxes = []
    if isinstance(boxes, list):
        for box in boxes:
            if isinstance(box, list) and len(box) == 4:
                try:
                    clean_boxes.append([float(v) for v in box])
                except Exception:
                    pass
    data["boxes"] = clean_boxes
    return data


def boxes_to_mask(boxes: list[list[float]], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1i = int(max(0, min(width, round(x1 / 1000.0 * width))))
        x2i = int(max(0, min(width, round(x2 / 1000.0 * width))))
        y1i = int(max(0, min(height, round(y1 / 1000.0 * height))))
        y2i = int(max(0, min(height, round(y2 / 1000.0 * height))))
        if x2i > x1i and y2i > y1i:
            mask[y1i:y2i, x1i:x2i] = 1
    return mask


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
        args.output_dir = args.project_root / "12_LocatingAnything" / "grounding_eval"
    if args.results_csv is None:
        args.results_csv = args.project_root / "results" / "locateanything_grounding_results.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    exact_rounds = args.rounds_per_dataset is not None
    max_samples = args.rounds_per_dataset or args.max_samples_per_dataset
    samples = collect_samples(args.binary_root, args.datasets, args.split, max_samples, args.seed, exact_rounds)
    model, processor, tokenizer = load_model(args.model_id, args.device)
    rows = []
    jsonl_path = args.output_dir / "locateanything_raw_outputs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for idx, sample in enumerate(samples, start=1):
            image = Image.open(sample.image).convert("RGB")
            raw = run_one(model, processor, tokenizer, sample.image, args.max_new_tokens, args.device)
            parsed = parse_json_response(raw)
            boxes = parsed.get("boxes", [])
            pred_mask = boxes_to_mask(boxes if isinstance(boxes, list) else [], image.size)
            metrics = mask_metrics(pred_mask, sample.mask, image.size)
            row = {
                "dataset": sample.dataset,
                "split": sample.split,
                "round_index": sample.round_index,
                "sample": sample.name,
                "model": args.model_id,
                "box_count": len(boxes) if isinstance(boxes, list) else 0,
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
