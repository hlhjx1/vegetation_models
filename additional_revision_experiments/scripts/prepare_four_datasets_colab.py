#!/usr/bin/env python3
"""Download and preprocess the four vegetation datasets on Colab.

Datasets:
1. Zijinshan: copied from Drive datasets/2024-seg.
2. LoveDA: downloaded from Zenodo and converted to binary vegetation masks.
3. Potsdam: downloaded from Kaggle and converted to binary vegetation patches.
4. Vaihingen: downloaded from Kaggle and converted to binary vegetation patches.

Outputs are VOC-like binary segmentation folders:
  /content/binary/{zijinshan,loveda,potsdam,vaihingen}/JPEGImages
  /content/binary/{zijinshan,loveda,potsdam,vaihingen}/SegmentationClass
  /content/binary/{dataset}/ImageSets/Segmentation/default.txt

This script prepares data only. It does not train models.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare four binary vegetation segmentation datasets.")
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive"))
    parser.add_argument("--project-root", type=Path, default=Path("/content/vegetation_models_v2"))
    parser.add_argument("--work-root", type=Path, default=Path("/content"))
    parser.add_argument("--binary-root", type=Path, default=Path("/content/binary"))
    parser.add_argument("--max-loveda", type=int, default=500)
    parser.add_argument("--max-potsdam", type=int, default=400)
    parser.add_argument("--max-vaihingen", type=int, default=400)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--min-veg-ratio", type=float, default=0.05)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--copy-to-drive", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_default_split(dataset_dir: Path) -> None:
    image_dir = dataset_dir / "JPEGImages"
    split_dir = ensure(dataset_dir / "ImageSets" / "Segmentation")
    names = sorted(path.stem for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    (split_dir / "default.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def write_labelmap(dataset_dir: Path) -> None:
    (dataset_dir / "labelmap.txt").write_text(
        "# label:color_rgb:parts:actions\nbackground:0,0,0::\nvegetation:144,32,192::\n",
        encoding="utf-8",
    )


def dataset_dirs(root: Path, name: str) -> tuple[Path, Path, Path]:
    ds_dir = ensure(root / name)
    img_dir = ensure(ds_dir / "JPEGImages")
    mask_dir = ensure(ds_dir / "SegmentationClass")
    return ds_dir, img_dir, mask_dir


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8)).save(path)


def copy_zijinshan(args: argparse.Namespace) -> int:
    source_candidates = [
        args.drive_root / "datasets" / "2024-seg",
        args.drive_root / "vegetation_models_v2" / "datasets" / "2024-seg",
        args.project_root / "datasets" / "2024-seg",
    ]
    source = next((path for path in source_candidates if (path / "JPEGImages").is_dir()), None)
    if source is None:
        print("WARNING: Zijinshan source dataset not found; checked:")
        for path in source_candidates:
            print(f"  - {path}")
        return 0
    ds_dir = ensure(args.binary_root / "zijinshan")
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    shutil.copytree(source, ds_dir)
    write_default_split(ds_dir)
    write_labelmap(ds_dir)
    count = len(list((ds_dir / "JPEGImages").glob("*")))
    print(f"Zijinshan copied: {count} images -> {ds_dir}")
    return count


def download_loveda(args: argparse.Namespace) -> None:
    loveda_root = ensure(args.work_root / "loveda")
    train_zip = loveda_root / "Train.zip"
    val_zip = loveda_root / "Val.zip"
    if (loveda_root / "Train").is_dir() and (loveda_root / "Val").is_dir():
        print("LoveDA already extracted, skip download.")
        return
    run(["wget", "-q", "--show-progress", "https://zenodo.org/records/5706578/files/Train.zip?download=1", "-O", str(train_zip)])
    run(["wget", "-q", "--show-progress", "https://zenodo.org/records/5706578/files/Val.zip?download=1", "-O", str(val_zip)])
    run(["unzip", "-q", str(train_zip), "-d", str(loveda_root)])
    run(["unzip", "-q", str(val_zip), "-d", str(loveda_root)])
    train_zip.unlink(missing_ok=True)
    val_zip.unlink(missing_ok=True)


def download_isprs(args: argparse.Namespace) -> None:
    isprs_root = ensure(args.work_root / "isprs")
    if any(isprs_root.iterdir()):
        print("ISPRS directory is not empty, skip Kaggle download.")
        return
    kaggle_json = args.work_root / "kaggle.json"
    if not kaggle_json.exists() and os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        kaggle_json.write_text(
            json.dumps({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]}),
            encoding="utf-8",
        )
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "Missing /content/kaggle.json. Upload it in Colab or set KAGGLE_USERNAME and KAGGLE_KEY first."
        )
    os.environ["KAGGLE_CONFIG_DIR"] = str(args.work_root)
    kaggle_json.chmod(0o600)
    run(["kaggle", "datasets", "download", "-d", "bkfateam/potsdamvaihingen", "-p", str(isprs_root), "--unzip"])


def preprocess_loveda(args: argparse.Namespace) -> int:
    ds_dir, out_img, out_mask = dataset_dirs(args.binary_root, "loveda")
    for path in [out_img, out_mask]:
        shutil.rmtree(path, ignore_errors=True)
        ensure(path)
    total = 0
    for split in ["Train", "Val"]:
        for scene in ["Urban", "Rural"]:
            if total >= args.max_loveda:
                break
            img_dir = None
            for candidate in [
                args.work_root / "loveda" / split / scene / "images_png",
                args.work_root / "loveda" / split / scene / "images",
            ]:
                if candidate.exists():
                    img_dir = candidate
                    break
            if img_dir is None:
                print(f"Skip LoveDA {split}/{scene}: image dir not found")
                continue
            mask_dir = Path(str(img_dir).replace("images_png", "masks_png").replace("/images", "/masks_png"))
            if not mask_dir.exists():
                print(f"Skip LoveDA {split}/{scene}: mask dir not found")
                continue
            for image_path in sorted(img_dir.iterdir()):
                if total >= args.max_loveda:
                    break
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                    continue
                mask_path = mask_dir / f"{image_path.stem}.png"
                if not mask_path.exists():
                    continue
                img_arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
                mask_arr = np.asarray(Image.open(mask_path))
                binary = np.zeros(mask_arr.shape[:2], dtype=np.uint8)
                # LoveDA labels: 6 forest, 7 agriculture.
                binary[(mask_arr == 6) | (mask_arr == 7)] = 255
                stem = f"loveda_{total:04d}"
                Image.fromarray(img_arr).save(out_img / f"{stem}.png")
                save_mask(binary, out_mask / f"{stem}.png")
                total += 1
    write_default_split(ds_dir)
    write_labelmap(ds_dir)
    print(f"LoveDA preprocessed: {total} images -> {ds_dir}")
    return total


def save_patches(
    image: np.ndarray,
    binary: np.ndarray,
    out_img: Path,
    out_mask: Path,
    prefix: str,
    start: int,
    max_total: int,
    patch: int,
    min_veg_ratio: float,
) -> int:
    height, width = image.shape[:2]
    saved = 0
    for y in range(0, height - patch + 1, patch):
        for x in range(0, width - patch + 1, patch):
            if start + saved >= max_total:
                return saved
            image_patch = image[y : y + patch, x : x + patch]
            mask_patch = binary[y : y + patch, x : x + patch]
            if float((mask_patch == 255).sum()) / float(patch * patch) < min_veg_ratio:
                continue
            stem = f"{prefix}_{start + saved:04d}"
            Image.fromarray(image_patch.astype(np.uint8)).save(out_img / f"{stem}.png")
            save_mask(mask_patch, out_mask / f"{stem}.png")
            saved += 1
    return saved


def find_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exist:\n" + "\n".join(str(path) for path in paths))


def preprocess_potsdam(args: argparse.Namespace) -> int:
    isprs = args.work_root / "isprs"
    img_root = find_existing([isprs / "3_Ortho_IRRG" / "3_Ortho_IRRG"])
    mask_root = find_existing([isprs / "5_Labels_for_participants_no_Boundary" / "5_Labels_for_participants_no_Boundary"])
    ds_dir, out_img, out_mask = dataset_dirs(args.binary_root, "potsdam")
    for path in [out_img, out_mask]:
        shutil.rmtree(path, ignore_errors=True)
        ensure(path)
    mask_files = sorted(path for path in mask_root.iterdir() if path.suffix.lower() == ".tif")
    mask_dict = {path.name.replace("_label_noBoundary.tif", ""): path for path in mask_files}
    total = 0
    for image_path in sorted(path for path in img_root.iterdir() if path.suffix.lower() == ".tif"):
        if total >= args.max_potsdam:
            break
        key = image_path.name.replace("_IRRG.tif", "")
        mask_path = mask_dict.get(key)
        if mask_path is None:
            continue
        image = np.asarray(Image.open(image_path), dtype=np.uint8)[:, :, :3]
        mask = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        binary = np.zeros(mask.shape[:2], dtype=np.uint8)
        # ISPRS: tree and low vegetation.
        binary[np.all(mask == [0, 255, 0], axis=2)] = 255
        binary[np.all(mask == [0, 255, 255], axis=2)] = 255
        total += save_patches(
            image, binary, out_img, out_mask, "potsdam", total, args.max_potsdam, args.patch_size, args.min_veg_ratio
        )
    write_default_split(ds_dir)
    write_labelmap(ds_dir)
    print(f"Potsdam preprocessed: {total} patches -> {ds_dir}")
    return total


def preprocess_vaihingen(args: argparse.Namespace) -> int:
    isprs = args.work_root / "isprs"
    img_root = find_existing([isprs / "isprs_semantic_labeling_vaihingen" / "top"])
    mask_root = find_existing([isprs / "isprs_semantic_labeling_vaihingen" / "gts_for_participants"])
    ds_dir, out_img, out_mask = dataset_dirs(args.binary_root, "vaihingen")
    for path in [out_img, out_mask]:
        shutil.rmtree(path, ignore_errors=True)
        ensure(path)
    mask_files = {path.name for path in mask_root.iterdir()}
    total = 0
    for image_path in sorted(path for path in img_root.iterdir() if path.suffix.lower() == ".tif"):
        if total >= args.max_vaihingen:
            break
        if image_path.name not in mask_files:
            continue
        mask_path = mask_root / image_path.name
        image = np.asarray(Image.open(image_path), dtype=np.uint8)[:, :, :3]
        mask = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        binary = np.zeros(mask.shape[:2], dtype=np.uint8)
        # ISPRS: tree and low vegetation.
        binary[np.all(mask == [0, 255, 0], axis=2)] = 255
        binary[np.all(mask == [0, 255, 255], axis=2)] = 255
        total += save_patches(
            image,
            binary,
            out_img,
            out_mask,
            "vaihingen",
            total,
            args.max_vaihingen,
            args.patch_size,
            args.min_veg_ratio,
        )
    write_default_split(ds_dir)
    write_labelmap(ds_dir)
    print(f"Vaihingen preprocessed: {total} patches -> {ds_dir}")
    return total


def copy_binary_to_drive(args: argparse.Namespace) -> None:
    target = args.drive_root / "vegetation_models_v2" / "datasets_binary"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(args.binary_root, target)
    print(f"Copied binary datasets to Drive: {target}")


def main() -> int:
    args = parse_args()
    ensure(args.project_root)
    ensure(args.binary_root)
    if not args.skip_download:
        download_loveda(args)
        download_isprs(args)
    counts = {
        "zijinshan": copy_zijinshan(args),
        "loveda": preprocess_loveda(args),
        "potsdam": preprocess_potsdam(args),
        "vaihingen": preprocess_vaihingen(args),
    }
    manifest = {
        "binary_root": str(args.binary_root),
        "counts": counts,
        "class_protocol": {
            "background": 0,
            "vegetation": 255,
            "LoveDA_vegetation_labels": [6, 7],
            "ISPRS_vegetation_colors_rgb": [[0, 255, 0], [0, 255, 255]],
        },
    }
    manifest_path = args.project_root / "results" / "four_dataset_prepare_manifest.json"
    ensure(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if args.copy_to_drive:
        copy_binary_to_drive(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
