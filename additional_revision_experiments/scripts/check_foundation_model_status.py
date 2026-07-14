#!/usr/bin/env python3
"""Check local status for the added foundation-model experiments.

This is a lightweight preflight script. It does not download weights and does
not train models. Paths are checked across the project root, the default local
cache, and common Colab/AutoDL cache locations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DINOv2/DINOv3/QwenVL/LocateAnything readiness.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--storage-root", type=Path, default=None)
    parser.add_argument("--write-json", type=Path, default=None)
    return parser.parse_args()


def has_any_file(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def existing(paths: list[Path]) -> list[str]:
    found: list[str] = []
    for path in paths:
        if path.is_file() or has_any_file(path):
            found.append(str(path))
    return found


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    storage_root = (
        args.storage_root.resolve()
        if args.storage_root
        else Path(os.environ.get("VEGETATION_FOUNDATION_ROOT", project_root)).resolve()
    )

    candidates = {
        "DINOv2": [
            project_root / "_cache" / "torch_cache" / "hub" / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth",
            storage_root / "_cache" / "torch_cache" / "hub" / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth",
            Path("/content/vegetation_foundation_models/_cache/torch_cache/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"),
            Path("/root/autodl-tmp/vegetation_foundation_models/_cache/torch_cache/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"),
        ],
        "DINOv3": [
            project_root / "10_DINOv3" / "weights",
            storage_root / "10_DINOv3" / "weights",
            Path("/content/vegetation_foundation_models/10_DINOv3/weights"),
            Path("/root/autodl-tmp/vegetation_foundation_models/10_DINOv3/weights"),
        ],
        "QwenVL": [
            project_root / "11_QwenVL" / "weights",
            storage_root / "11_QwenVL" / "weights",
            Path("/content/vegetation_foundation_models/11_QwenVL/weights"),
            Path("/root/autodl-tmp/vegetation_foundation_models/11_QwenVL/weights"),
        ],
        "LocateAnything": [
            project_root / "12_LocatingAnything" / "weights",
            storage_root / "12_LocatingAnything" / "weights",
            Path("/content/vegetation_foundation_models/12_LocatingAnything/weights"),
            Path("/root/autodl-tmp/vegetation_foundation_models/12_LocatingAnything/weights"),
        ],
    }

    report: dict[str, dict[str, object]] = {}
    for name, paths in candidates.items():
        present = existing(paths)
        if name == "DINOv2":
            ready = bool(present)
            note = "Ready for frozen-backbone smoke training if torch.hub code is also cached."
        elif name == "DINOv3":
            ready = any(has_any_file(path) for path in paths)
            note = "Gated HF access is required; smoke entry should skip gracefully until weights exist."
        elif name == "QwenVL":
            ready = any(has_any_file(path) for path in paths)
            note = "Use prompt-based recognition/localization smoke test, not dense segmentation fine-tuning."
        else:
            ready = any(has_any_file(path) for path in paths)
            note = "Use prompt-based grounding smoke test; box-mask metrics are only supplemental."
        report[name] = {
            "ready": ready,
            "present_paths": present,
            "checked_paths": [str(path) for path in paths],
            "note": note,
        }

    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for name, item in report.items():
        status = "READY" if item["ready"] else "MISSING"
        print(f"[{status}] {name}")
        for path in item["present_paths"]:
            print(f"  found: {path}")
        print(f"  note: {item['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
