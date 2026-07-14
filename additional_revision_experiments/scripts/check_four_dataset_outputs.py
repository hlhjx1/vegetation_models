#!/usr/bin/env python3
"""Check four prepared VOC-like binary vegetation datasets and manifest.

This is a read-only validation helper for Colab/Drive outputs. It verifies
file counts, image-mask pairs, split files, labelmap files, and optional
manifest counts. For Zijinshan, it also checks the temporal protocol:
train.txt = 2022+2023, test.txt = 2024.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_NAMES = ("zijinshan", "loveda", "potsdam", "vaihingen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate four prepared binary segmentation datasets.")
    parser.add_argument("--project-root", type=Path, default=Path("/content/vegetation_models_v2"))
    parser.add_argument("--binary-root", type=Path, default=Path("/content/binary"))
    parser.add_argument("--drive-project-root", type=Path, default=Path("/content/drive/MyDrive/vegetation_models_v2"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--write-json", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object] | None:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_manifests(args: argparse.Namespace) -> tuple[list[str], dict[str, object]]:
    candidates = []
    if args.manifest:
        candidates.append(args.manifest)
    candidates.extend(
        [
            args.project_root / "results" / "four_dataset_prepare_manifest.json",
            args.project_root / "results" / "zijinshan_temporal_manifest.json",
            args.drive_project_root / "results" / "four_dataset_prepare_manifest.json",
            args.drive_project_root / "results" / "zijinshan_temporal_manifest.json",
            args.drive_project_root / "datasets_binary" / "four_dataset_prepare_manifest.json",
        ]
    )
    loaded_paths: list[str] = []
    combined: dict[str, object] = {"counts": {}}
    for path in candidates:
        manifest = load_json(path)
        if manifest is None:
            continue
        loaded_paths.append(str(path))
        counts = manifest.get("counts", {})
        if isinstance(counts, dict):
            combined_counts = combined.setdefault("counts", {})
            if isinstance(combined_counts, dict):
                for key, value in counts.items():
                    combined_counts[str(key)] = int(value)
        if manifest.get("dataset") == "zijinshan":
            total_count = manifest.get("total_count")
            output_root = manifest.get("output_root")
            if total_count is not None:
                combined_counts = combined.setdefault("counts", {})
                if isinstance(combined_counts, dict):
                    combined_counts["zijinshan"] = int(total_count)
            if output_root:
                combined["zijinshan_output_root"] = str(output_root)
        binary_root = manifest.get("binary_root")
        if binary_root:
            combined["binary_root"] = str(binary_root)
    return loaded_paths, combined


def list_stems(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {
        item.stem
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    }


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_dataset(root: Path, name: str, expected_count: int | None) -> dict[str, object]:
    ds = root / name
    images = list_stems(ds / "JPEGImages")
    masks = list_stems(ds / "SegmentationClass")
    split_dir = ds / "ImageSets" / "Segmentation"
    split = read_split(split_dir / "default.txt")
    train_split = read_split(split_dir / "train.txt")
    test_split = read_split(split_dir / "test.txt")
    split_set = set(split)
    train_set = set(train_split)
    test_set = set(test_split)
    paired = images & masks
    missing_masks = sorted(images - masks)
    missing_images = sorted(masks - images)
    split_missing_pairs = sorted(split_set - paired)
    pair_not_in_split = sorted(paired - split_set)
    temporal_ok = None
    temporal_errors: list[str] = []
    if name == "zijinshan":
        temporal_ok = True
        if not train_split:
            temporal_errors.append("missing_or_empty_train_txt")
        if not test_split:
            temporal_errors.append("missing_or_empty_test_txt")
        if train_set & test_set:
            temporal_errors.append("train_test_overlap")
        if train_set - paired:
            temporal_errors.append("train_entries_missing_pairs")
        if test_set - paired:
            temporal_errors.append("test_entries_missing_pairs")
        if train_split and not all(item.startswith(("2022_", "2023_")) for item in train_split):
            temporal_errors.append("train_txt_contains_non_2022_2023_prefix")
        if test_split and not all(item.startswith("2024_") for item in test_split):
            temporal_errors.append("test_txt_contains_non_2024_prefix")
        if split and set(split) != train_set | test_set:
            temporal_errors.append("default_txt_not_equal_train_plus_test")
        temporal_ok = not temporal_errors
    ok = (
        ds.is_dir()
        and (ds / "labelmap.txt").is_file()
        and len(images) > 0
        and len(images) == len(masks) == len(split) == len(paired)
        and not missing_masks
        and not missing_images
        and not split_missing_pairs
        and not pair_not_in_split
        and (expected_count is None or len(paired) == expected_count)
        and (temporal_ok is not False)
    )
    return {
        "dataset": name,
        "root": str(ds),
        "exists": ds.is_dir(),
        "image_count": len(images),
        "mask_count": len(masks),
        "split_count": len(split),
        "train_split_count": len(train_split),
        "test_split_count": len(test_split),
        "paired_count": len(paired),
        "expected_count": expected_count,
        "labelmap_exists": (ds / "labelmap.txt").is_file(),
        "temporal_protocol_ok": temporal_ok,
        "temporal_protocol_errors": temporal_errors,
        "missing_masks_preview": missing_masks[:10],
        "missing_images_preview": missing_images[:10],
        "split_missing_pairs_preview": split_missing_pairs[:10],
        "pair_not_in_split_preview": pair_not_in_split[:10],
        "ok": ok,
    }


def main() -> int:
    args = parse_args()
    manifest_paths, manifest = load_manifests(args)
    manifest_counts = {}
    if manifest:
        counts = manifest.get("counts", {})
        if isinstance(counts, dict):
            manifest_counts = {str(key): int(value) for key, value in counts.items()}
        manifest_binary_root = manifest.get("binary_root")
        if manifest.get("zijinshan_output_root"):
            manifest_binary_root = str(Path(str(manifest.get("zijinshan_output_root"))).parent)
        if manifest_binary_root and args.binary_root == Path("/content/binary"):
            args.binary_root = Path(str(manifest_binary_root))

    report = {
        "manifest_paths": manifest_paths,
        "binary_root": str(args.binary_root),
        "datasets": [
            check_dataset(args.binary_root, name, manifest_counts.get(name))
            for name in DATASET_NAMES
        ],
    }
    report["all_ok"] = all(bool(item["ok"]) for item in report["datasets"])

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote validation report: {args.write_json}")
    return 0 if report["all_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
