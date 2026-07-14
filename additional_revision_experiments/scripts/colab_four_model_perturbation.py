#!/usr/bin/env python3
"""Colab perturbation runner for DINOv2, DINOv3, and QwenVL.

The perturbation settings intentionally match the old eight-model robustness
script `scripts/run_robustness_perturbation_eval.py`:

- brightness: 0.70, 0.85, 1.15, 1.30
- contrast: 0.75, 0.90, 1.10, 1.25
- gaussian_blur: kernel_3, kernel_5, kernel_7
- shadow_illumination: mild, moderate, strong

LocateAnything is intentionally excluded here because it needs a separate
dependency environment.

Outputs:
- results/four_model_robustness_colab.csv
- results/four_model_prompt_detail_colab.csv
- results/four_model_colab_path_report.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch


DATASETS = ("zijinshan",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Colab perturbation tests for DINOv2/DINOv3/QwenVL.")
    parser.add_argument("--project-root", type=Path, default=Path("/content/vegetation_models_v2"))
    parser.add_argument("--binary-root", type=Path, default=Path("/content/binary"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--split", choices=["auto", "test", "val", "default"], default="auto")
    parser.add_argument("--rounds-per-dataset", type=int, default=100, help="QwenVL prompt rounds per dataset.")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--prompt-detail-csv", type=Path, default=None)
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        print(f"No rows to write for {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def main() -> int:
    args = parse_args()
    results_dir = args.project_root / "results"
    args.output_csv = args.output_csv or (results_dir / "four_model_robustness_colab.csv")
    args.prompt_detail_csv = args.prompt_detail_csv or (results_dir / "four_model_prompt_detail_colab.csv")
    args.perturbed_image_root = args.perturbed_image_root or (results_dir / "four_model_perturbed_prompt_images")

    tool = import_file(args.project_root / "scripts/colab_four_model_efficiency_robustness.py", "colab_four_tool_perturb")
    tool.configure_torch(args)

    path_report = tool.scan_paths(args.project_root)
    report_path = results_dir / "four_model_colab_path_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(path_report, indent=2), encoding="utf-8")
    print(json.dumps(path_report, indent=2))

    print("Perturbation settings are exactly aligned with the old eight-model script:")
    for condition in tool.conditions():
        print(f"  - {condition.perturbation}: {condition.strength}")

    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    if not args.skip_dino:
        rows.extend(tool.run_dino_robustness(args))
    if not args.skip_qwen:
        qwen_summary, qwen_detail = tool.run_qwen_prompt(args, args.perturbed_image_root)
        rows.extend(qwen_summary)
        detail_rows.extend(qwen_detail)

    write_csv(args.output_csv, rows)
    write_csv(args.prompt_detail_csv, detail_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
