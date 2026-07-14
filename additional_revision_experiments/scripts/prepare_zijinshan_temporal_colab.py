#!/usr/bin/env python3
"""Prepare Zijinshan temporal split for Colab experiments.

Protocol:
  train: 2022-seg + 2023-seg
  test:  2024-seg

The output is still VOC-like, but keeps explicit temporal split files:
  /content/binary/zijinshan/JPEGImages
  /content/binary/zijinshan/SegmentationClass
  /content/binary/zijinshan/ImageSets/Segmentation/train.txt
  /content/binary/zijinshan/ImageSets/Segmentation/test.txt
  /content/binary/zijinshan/ImageSets/Segmentation/default.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


YEARS = ("2022-seg", "2023-seg", "2024-seg")
TRAIN_YEARS = ("2022-seg", "2023-seg")
TEST_YEARS = ("2024-seg",)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy all Zijinshan yearly datasets into one temporal split dataset.")
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive"))
    parser.add_argument("--project-root", type=Path, default=Path("/content/vegetation_models_v2"))
    parser.add_argument("--binary-root", type=Path, default=Path("/content/binary"))
    parser.add_argument("--copy-to-drive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_dataset_root(args: argparse.Namespace, year: str) -> Path:
    candidates = [
        args.drive_root / "datasets" / year,
        args.drive_root / "vegetation_models_v2" / "datasets" / year,
        args.project_root / "datasets" / year,
    ]
    for path in candidates:
        if (path / "JPEGImages").is_dir() and (path / "SegmentationClass").is_dir():
            return path
    raise FileNotFoundError("Missing Zijinshan year dataset. Checked:\n" + "\n".join(str(p) for p in candidates))


def find_pair(image_dir: Path, mask_dir: Path, stem: str) -> tuple[Path, Path] | None:
    image_path = None
    mask_path = None
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            image_path = candidate
            break
    for suffix in IMAGE_SUFFIXES:
        candidate = mask_dir / f"{stem}{suffix}"
        if candidate.is_file():
            mask_path = candidate
            break
    if image_path is None or mask_path is None:
        return None
    return image_path, mask_path


def stems_from_year(year_root: Path) -> list[str]:
    split_path = year_root / "ImageSets" / "Segmentation" / "default.txt"
    if split_path.is_file():
        stems = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if stems:
            return stems
    return sorted(path.stem for path in (year_root / "JPEGImages").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def write_labelmap(dataset_dir: Path) -> None:
    (dataset_dir / "labelmap.txt").write_text(
        "# label:color_rgb:parts:actions\nbackground:0,0,0::\nvegetation:144,32,192::\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    out_dir = args.binary_root / "zijinshan"
    image_out = out_dir / "JPEGImages"
    mask_out = out_dir / "SegmentationClass"
    split_out = out_dir / "ImageSets" / "Segmentation"

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists. Add --overwrite to replace it.")

    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    split_out.mkdir(parents=True, exist_ok=True)

    train_names: list[str] = []
    test_names: list[str] = []
    counts: dict[str, int] = {}

    for year in YEARS:
        year_root = find_dataset_root(args, year)
        image_dir = year_root / "JPEGImages"
        mask_dir = year_root / "SegmentationClass"
        copied = 0
        year_prefix = year.replace("-seg", "")
        for stem in stems_from_year(year_root):
            pair = find_pair(image_dir, mask_dir, stem)
            if pair is None:
                print(f"WARNING: missing image/mask pair for {year}/{stem}")
                continue
            image_path, mask_path = pair
            out_stem = f"{year_prefix}_{stem}"
            shutil.copy2(image_path, image_out / f"{out_stem}.png")
            shutil.copy2(mask_path, mask_out / f"{out_stem}.png")
            if year in TRAIN_YEARS:
                train_names.append(out_stem)
            elif year in TEST_YEARS:
                test_names.append(out_stem)
            copied += 1
        counts[year] = copied
        print(f"{year}: copied {copied} pairs from {year_root}")

    all_names = train_names + test_names
    (split_out / "train.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
    (split_out / "test.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")
    (split_out / "default.txt").write_text("\n".join(all_names) + "\n", encoding="utf-8")
    write_labelmap(out_dir)

    manifest = {
        "dataset": "zijinshan",
        "output_root": str(out_dir),
        "protocol": "temporal_split",
        "train_years": list(TRAIN_YEARS),
        "test_years": list(TEST_YEARS),
        "counts_by_year": counts,
        "train_count": len(train_names),
        "test_count": len(test_names),
        "total_count": len(all_names),
    }
    manifest_path = args.project_root / "results" / "zijinshan_temporal_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    if args.copy_to_drive:
        drive_target = args.drive_root / "vegetation_models_v2" / "datasets_binary" / "zijinshan"
        if drive_target.exists():
            shutil.rmtree(drive_target)
        shutil.copytree(out_dir, drive_target)
        drive_manifest = args.drive_root / "vegetation_models_v2" / "results" / "zijinshan_temporal_manifest.json"
        drive_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, drive_manifest)
        print(f"Copied Zijinshan temporal dataset to Drive: {drive_target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
