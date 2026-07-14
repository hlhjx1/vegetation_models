#!/usr/bin/env python3
"""Recompute QwenVL prompt-eval CSV metrics from saved raw JSONL outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from run_qwenvl_vegetation_prompt_eval import (
    boxes_to_mask,
    existing_file,
    mask_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute QwenVL prompt-eval metrics from raw JSONL.")
    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        default=Path("/content/vegetation_models_v2/11_QwenVL/prompt_eval/qwenvl_raw_outputs.jsonl"),
    )
    parser.add_argument("--binary-root", type=Path, default=Path("/content/binary"))
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("/content/vegetation_models_v2/results/qwenvl_prompt_eval_results.csv"),
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=["auto", "normalized1000", "pixel"],
        default="auto",
        help="Auto treats boxes within image dimensions as pixel coordinates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with args.raw_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            obj = json.loads(line)
            row = dict(obj["row"])
            dataset = str(row["dataset"])
            sample = str(row["sample"])
            root = args.binary_root / dataset
            image_path = existing_file(root / "JPEGImages", sample)
            mask_path = existing_file(root / "SegmentationClass", sample)
            if image_path is None or mask_path is None:
                raise FileNotFoundError(
                    f"Missing image or mask for line {line_number}: dataset={dataset}, sample={sample}"
                )
            image = Image.open(image_path).convert("RGB")
            boxes = obj.get("parsed", {}).get("boxes", [])
            if not isinstance(boxes, list):
                boxes = []
            pred_mask, coordinate_modes = boxes_to_mask(boxes, image.size, args.coordinate_mode)
            metrics = mask_metrics(pred_mask, mask_path, image.size)
            row["coordinate_mode"] = args.coordinate_mode
            row["pixel_box_count"] = coordinate_modes["pixel"]
            row["normalized1000_box_count"] = coordinate_modes["normalized1000"]
            row.update(metrics)
            rows.append(row)
    with args.results_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rows: {len(rows)}")
    print(f"Wrote CSV: {args.results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
