#!/usr/bin/env python3
"""Colab efficiency runner for DINOv2 and DINOv3.

LocateAnything is intentionally excluded here because it needs a separate
dependency environment.

Default protocol matches the old efficiency benchmark:
- batch size 1
- 10 warm-up iterations
- 100 timed forward passes
- CUDA synchronization before/after timing
- peak GPU memory during measured forward pass
- no image loading, visualization, or CPU post-processing
- FLOPs as 2 x THOP MACs when available

QwenVL can be enabled with --include-qwen, but that is supplemental
prompt-to-box generation latency, not the forward-only benchmark.

Outputs:
- results/four_model_efficiency_colab.csv
- results/four_model_colab_path_report.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


DATASETS = ("zijinshan",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Colab efficiency tests for DINOv2/DINOv3/QwenVL.")
    parser.add_argument("--project-root", type=Path, default=Path("/content/vegetation_models_v2"))
    parser.add_argument("--binary-root", type=Path, default=Path("/content/binary"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--split", choices=["auto", "test", "val", "default"], default="auto")
    parser.add_argument("--rounds-per-dataset", type=int, default=20, help="Clean QwenVL samples per dataset for latency.")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--no-flops", action="store_true")
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true", help="Deprecated alias; Qwen is skipped unless --include-qwen is set.")
    parser.add_argument(
        "--include-qwen",
        action="store_true",
        help="Also run supplemental QwenVL prompt-to-box latency. This is not part of the forward-only efficiency benchmark.",
    )
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-csv", type=Path, default=None)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def estimate_qwen_prefill_flops_g(qwen, model, processor, process_vision_info, image_path: Path) -> tuple[object, str]:
    """Estimate QwenVL prompt-image prefill forward FLOPs.

    This is not equivalent to dense segmentation FLOPs and does not include the
    full autoregressive generation loop. It is still useful as a reproducible
    computational-cost proxy for the fixed prompt and image.
    """
    try:
        messages = qwen.prompt_for_image(image_path)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        if hasattr(model, "floating_point_ops"):
            flops = model.floating_point_ops(inputs)
            if flops:
                return round(float(flops) / 1e9, 3), "Estimated QwenVL prefill forward FLOPs from transformers `floating_point_ops`; generation-loop FLOPs are not included."

        profiler_kwargs = {
            "activities": [torch.profiler.ProfilerActivity.CPU],
            "with_flops": True,
            "record_shapes": False,
            "profile_memory": False,
        }
        if torch.cuda.is_available():
            profiler_kwargs["activities"].append(torch.profiler.ProfilerActivity.CUDA)
            torch.cuda.synchronize()
        model.eval()
        with torch.no_grad():
            with torch.profiler.profile(**profiler_kwargs) as prof:
                _ = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        flops = sum(event.flops for event in prof.key_averages() if getattr(event, "flops", 0))
        if flops:
            return round(float(flops) / 1e9, 3), "Estimated QwenVL prefill forward FLOPs from `torch.profiler` with `with_flops=True`; generation-loop FLOPs are not included."
        return "pending_profiler_no_flops", "Torch profiler ran but did not report FLOPs for this QwenVL forward; generation FLOPs remain prompt/output dependent."
    except Exception as exc:
        return "pending_estimation_failed", f"QwenVL FLOPs estimate failed: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"


def qwen_input_size_label(samples) -> str:
    sizes = []
    for sample in samples:
        try:
            from PIL import Image

            with Image.open(sample.image) as image:
                sizes.append(image.size)
        except Exception:
            pass
    unique = sorted(set(sizes))
    if not unique:
        return "original image size pending"
    if len(unique) == 1:
        width, height = unique[0]
        return f"{width}x{height} original"
    shown = ", ".join(f"{width}x{height}" for width, height in unique[:5])
    suffix = "" if len(unique) <= 5 else f", ... ({len(unique)} unique sizes)"
    return f"mixed original sizes: {shown}{suffix}"


def qwen_clean_latency(tool, args: argparse.Namespace) -> list[dict[str, object]]:
    qwen = import_file(args.project_root / "scripts/run_qwenvl_vegetation_prompt_eval.py", "colab_eff_qwen")
    load_args = argparse.Namespace(model_id=args.qwen_model_id, torch_dtype="auto", device=args.device)
    model, processor, process_vision_info = qwen.load_qwen(load_args)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    samples = tool.collect_samples(args.binary_root, args.datasets, args.split, args.rounds_per_dataset, args.seed)

    rows = []
    for dataset in args.datasets:
        dataset_samples = [sample for sample in samples if sample.dataset == dataset]
        input_size = qwen_input_size_label(dataset_samples)
        qwen_flops_g, qwen_flops_note = (
            estimate_qwen_prefill_flops_g(qwen, model, processor, process_vision_info, dataset_samples[0].image)
            if dataset_samples
            else ("pending_no_samples", "No QwenVL samples available for FLOPs estimation.")
        )
        latencies = []
        for idx, sample in enumerate(dataset_samples, start=1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = qwen.run_one(model, processor, process_vision_info, sample.image, args.max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(latency_ms)
            print(f"[qwen latency] {dataset} {idx}/{len(dataset_samples)} {sample.name}: {latency_ms:.1f} ms")
        mean_ms = float(np.mean(latencies)) if latencies else 0.0
        rows.append(
            {
                "model": args.qwen_model_id,
                "dataset_head": dataset,
                "task_type": "prompt-based inference",
                "params_M": params_m,
                "trainable_params_M": "N/A",
                "flops_G": qwen_flops_g,
                "flops_note": qwen_flops_note,
                "inference_time_ms": mean_ms,
                "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
                "gpu_memory_MB": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0,
                "input_size": input_size,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device,
                "notes": f"Clean prompt-to-box generation latency, n={len(dataset_samples)}, max_new_tokens={args.max_new_tokens}; FLOPs are QwenVL prefill estimate and are not directly comparable with dense segmentation forward FLOPs.",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    args.output_csv = args.output_csv or (args.project_root / "results/four_model_efficiency_colab.csv")
    tool = import_file(args.project_root / "scripts/colab_four_model_efficiency_robustness.py", "colab_four_tool_eff")

    path_report = tool.scan_paths(args.project_root)
    report_path = args.project_root / "results/four_model_colab_path_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(path_report, indent=2), encoding="utf-8")
    print(json.dumps(path_report, indent=2))

    rows: list[dict[str, object]] = []
    if not args.skip_dino:
        rows.extend(tool.run_dino_efficiency(args))
    if args.include_qwen and not args.skip_qwen:
        rows.extend(qwen_clean_latency(tool, args))
    write_csv(args.output_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
