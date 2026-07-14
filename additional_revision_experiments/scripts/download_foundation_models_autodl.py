#!/usr/bin/env python3
"""Download code and pretrained weights for the added foundation-model experiments.

This script is intended for AutoDL, for example:

  cd /root/vegetation_models_v2
  python scripts/download_foundation_models_autodl.py --install-deps

It only downloads repositories/checkpoints and writes a manifest. It does not train.
Default profile is storage-safe for one RTX PRO 6000 machine:
DINOv2 ViT-L, DINOv3 ViT-L/SAT, Qwen2.5-VL-3B, and LocateAnything-3B.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, desc=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            print(self.desc or "progress")
            return self

        def __exit__(self, *args):
            return False

        def update(self, n=1):
            return None

        @staticmethod
        def write(msg):
            print(msg)


def parse_args() -> argparse.Namespace:
    default_project_root = Path("/root/vegetation_models_v2") if os.name != "nt" else Path.cwd()
    if not default_project_root.exists():
        default_project_root = Path.cwd()
    default_storage_root = (
        Path("/root/autodl-tmp/vegetation_foundation_models")
        if os.name != "nt"
        else Path.cwd()
    )
    default_cache_root = default_storage_root / "_cache"

    parser = argparse.ArgumentParser(
        description="Download DINOv2, DINOv3, QwenVL, and LocateAnything code/weights for AutoDL."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root,
        help="Project root. Default: /root/vegetation_models_v2 if present, otherwise cwd.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=default_cache_root,
        help="Large cache root for Hugging Face and torch hub. Default on AutoDL: /root/autodl-tmp/vegetation_foundation_models/_cache.",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=default_storage_root,
        help="Where model code/weights are saved. Default on AutoDL: /root/autodl-tmp/vegetation_foundation_models.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", ""),
        help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com.",
    )
    parser.add_argument(
        "--qwen-model",
        default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
        help="QwenVL HF model id. Default: Qwen/Qwen2.5-VL-3B-Instruct for the storage-safe revision profile.",
    )
    parser.add_argument(
        "--dinov3-model",
        default=os.environ.get("DINOV3_MODEL", "facebook/dinov3-vitl16-pretrain-sat493m"),
        help="DINOv3 HF model id. Default uses the satellite-pretrained ViT-L checkpoint.",
    )
    parser.add_argument(
        "--dinov3-fallback-model",
        default=os.environ.get("DINOV3_FALLBACK_MODEL", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
        help="Fallback DINOv3 HF model id if the default checkpoint is unavailable.",
    )
    parser.add_argument(
        "--locate-model",
        default=os.environ.get("LOCATE_MODEL", "nvidia/LocateAnything-3B"),
        help="LocateAnything HF model id. Default remains nvidia/LocateAnything-3B because this is the common public checkpoint.",
    )
    parser.add_argument(
        "--dinov2-models",
        nargs="+",
        default=["dinov2_vitl14_reg"],
        help="DINOv2 torch hub model names to cache.",
    )
    parser.add_argument(
        "--download-large-dinov2",
        action="store_true",
        help="Also download dinov2_vitg14_reg. It is about 4.2 GB and can be slow from dl.fbaipublicfiles.com.",
    )
    parser.add_argument("--install-deps", action="store_true", help="Install download/runtime helper packages first.")
    parser.add_argument("--skip-dinov2", action="store_true")
    parser.add_argument("--skip-dinov3", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-locate", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading.")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    tqdm.write("$ " + " ".join(cmd))
    if dry_run:
        return
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tqdm.write(line)
    code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def ensure_dirs(project_root: Path, storage_root: Path, cache_root: Path) -> dict[str, Path]:
    dirs = {
        "storage": storage_root,
        "dinov2_code": storage_root / "9_DINOv2" / "code",
        "dinov2_weights": storage_root / "9_DINOv2" / "weights",
        "dinov3_code": storage_root / "10_DINOv3" / "code",
        "dinov3_weights": storage_root / "10_DINOv3" / "weights",
        "qwen_code": storage_root / "11_QwenVL" / "code",
        "qwen_weights": storage_root / "11_QwenVL" / "weights",
        "locate_code": storage_root / "12_LocatingAnything" / "code",
        "locate_weights": storage_root / "12_LocatingAnything" / "weights",
        "results": project_root / "results",
        "hf_cache": cache_root / "hf_cache",
        "torch_cache": cache_root / "torch_cache",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def install_deps(dry_run: bool) -> None:
    packages = [
        "pip",
        "setuptools",
        "wheel",
        "tqdm",
        "huggingface_hub[cli]",
        "transformers",
        "accelerate",
        "safetensors",
        "timm",
        "einops",
        "modelscope",
    ]
    run([sys.executable, "-m", "pip", "install", "-U", *packages], dry_run=dry_run)
    if shutil.which("git-lfs"):
        run(["git", "lfs", "install"], dry_run=dry_run)
    else:
        tqdm.write("git-lfs not found. GitHub code cloning still works; HF snapshot_download is used for weights.")


def clone_repo(url: str, target: Path, dry_run: bool) -> None:
    if target.exists() and any(target.iterdir()):
        tqdm.write(f"Skip clone, already exists: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", "--filter=blob:none", "--progress", url, str(target)], dry_run=dry_run)


def try_clone_repo(url: str, target: Path, dry_run: bool) -> dict[str, str]:
    try:
        clone_repo(url, target, dry_run)
        return {"status": "ok", "code_dir": str(target)}
    except Exception as exc:
        tqdm.write(f"Code clone failed, continue with weight download: {exc}")
        return {"status": "failed", "code_dir": str(target), "error": repr(exc)}


def snapshot_download(model_id: str, local_dir: Path, dry_run: bool) -> None:
    tqdm.write(f"HF download: {model_id} -> {local_dir}")
    if dry_run:
        return
    from huggingface_hub import snapshot_download as hf_snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        hf_snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception:
        endpoint = os.environ.get("HF_ENDPOINT", "")
        if not endpoint:
            raise
        tqdm.write(f"HF endpoint failed for {model_id}; retrying once without HF_ENDPOINT.")
        os.environ.pop("HF_ENDPOINT", None)
        try:
            hf_snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        finally:
            os.environ["HF_ENDPOINT"] = endpoint


def try_modelscope_download(model_id: str, local_dir: Path, dry_run: bool) -> None:
    tqdm.write(f"ModelScope fallback: {model_id} -> {local_dir}")
    if dry_run:
        return
    from modelscope import snapshot_download as ms_snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    ms_snapshot_download(model_id, local_dir=str(local_dir))


DINOv2_WEIGHT_URLS = {
    "dinov2_vitl14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    "dinov2_vitg14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
    "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    "dinov2_vitg14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
}


def checkpoint_name_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def resumable_download(url: str, target: Path, dry_run: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        mb = target.stat().st_size / 1024**2
        tqdm.write(f"Found existing/partial file, resume if possible: {target} ({mb:.1f} MB)")
    if dry_run:
        tqdm.write(f"Would download: {url} -> {target}")
        return
    if shutil.which("aria2c"):
        run(["aria2c", "-c", "-x", "8", "-s", "8", "-k", "1M", "-d", str(target.parent), "-o", target.name, url])
        return
    if shutil.which("wget"):
        run(["wget", "-c", url, "-O", str(target)])
        return

    headers = {}
    mode = "wb"
    downloaded = 0
    if target.exists():
        downloaded = target.stat().st_size
        headers["Range"] = f"bytes={downloaded}-"
        mode = "ab"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length", "0")) + downloaded
        with target.open(mode) as handle, tqdm(
            total=total, initial=downloaded, unit="B", unit_scale=True, desc=target.name
        ) as bar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bar.update(len(chunk))


def download_dinov2(models: Iterable[str], dry_run: bool) -> None:
    if dry_run:
        for name in models:
            tqdm.write(f"Would cache DINOv2 torch hub model: {name}")
        return
    import torch

    checkpoint_dir = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"
    for name in models:
        url = DINOv2_WEIGHT_URLS.get(name)
        if url:
            resumable_download(url, checkpoint_dir / checkpoint_name_from_url(url), dry_run)
        tqdm.write(f"Loading DINOv2 torch hub model: {name}")
        model = torch.hub.load("facebookresearch/dinov2", name)
        model.eval()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tqdm.write(f"DINOv2 cached OK: {name}")


def write_manifest(results_dir: Path, manifest: dict) -> Path:
    out = results_dir / "foundation_model_download_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    storage_root = args.storage_root.resolve()
    cache_root = args.cache_root.resolve()
    dirs = ensure_dirs(project_root, storage_root, cache_root)

    os.environ["HF_HOME"] = str(dirs["hf_cache"])
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(dirs["hf_cache"] / "hub")
    os.environ["TORCH_HOME"] = str(dirs["torch_cache"])
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    manifest: dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(project_root),
        "storage_root": str(storage_root),
        "cache_root": str(cache_root),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "tasks": {},
    }

    dinov2_models = list(args.dinov2_models)
    if args.download_large_dinov2 and "dinov2_vitg14_reg" not in dinov2_models:
        dinov2_models.append("dinov2_vitg14_reg")

    steps = []
    if args.install_deps:
        steps.append("install_deps")
    if not args.skip_dinov2:
        steps.append("dinov2")
    if not args.skip_dinov3:
        steps.append("dinov3")
    if not args.skip_qwen:
        steps.append("qwen")
    if not args.skip_locate:
        steps.append("locate")

    with tqdm(total=len(steps), desc="foundation model download") as bar:
        if args.install_deps:
            install_deps(args.dry_run)
            manifest["tasks"]["install_deps"] = {"status": "ok"}
            bar.update(1)

        if not args.skip_dinov2:
            task = {"status": "pending", "models": dinov2_models}
            try:
                task["code"] = try_clone_repo(
                    "https://github.com/facebookresearch/dinov2.git", dirs["dinov2_code"] / "dinov2", args.dry_run
                )
                download_dinov2(dinov2_models, args.dry_run)
                task["status"] = "ok"
                task["code_dir"] = str(dirs["dinov2_code"] / "dinov2")
                task["torch_cache"] = os.environ["TORCH_HOME"]
            except Exception as exc:
                task["status"] = "failed"
                task["error"] = repr(exc)
                tqdm.write(f"DINOv2 failed: {exc}")
            manifest["tasks"]["dinov2"] = task
            bar.update(1)

        if not args.skip_dinov3:
            task = {"status": "pending", "model": args.dinov3_model, "fallback_model": args.dinov3_fallback_model}
            try:
                task["code"] = try_clone_repo(
                    "https://github.com/facebookresearch/dinov3.git", dirs["dinov3_code"] / "dinov3", args.dry_run
                )
                local = dirs["dinov3_weights"] / args.dinov3_model.split("/")[-1]
                try:
                    snapshot_download(args.dinov3_model, local, args.dry_run)
                    task["downloaded_model"] = args.dinov3_model
                    task["weights_dir"] = str(local)
                except Exception as first_exc:
                    tqdm.write(f"DINOv3 primary checkpoint failed: {first_exc}")
                    fallback = dirs["dinov3_weights"] / args.dinov3_fallback_model.split("/")[-1]
                    snapshot_download(args.dinov3_fallback_model, fallback, args.dry_run)
                    task["downloaded_model"] = args.dinov3_fallback_model
                    task["weights_dir"] = str(fallback)
                task["status"] = "ok"
                task["code_dir"] = str(dirs["dinov3_code"] / "dinov3")
            except Exception as exc:
                task["status"] = "failed"
                task["error"] = repr(exc)
                tqdm.write(f"DINOv3 failed: {exc}")
            manifest["tasks"]["dinov3"] = task
            bar.update(1)

        if not args.skip_qwen:
            task = {"status": "pending", "model": args.qwen_model}
            try:
                task["code"] = try_clone_repo(
                    "https://github.com/QwenLM/Qwen2.5-VL.git", dirs["qwen_code"] / "Qwen2.5-VL", args.dry_run
                )
                local = dirs["qwen_weights"] / args.qwen_model.split("/")[-1]
                try:
                    snapshot_download(args.qwen_model, local, args.dry_run)
                except Exception as hf_exc:
                    tqdm.write(f"Qwen HF download failed: {hf_exc}")
                    try_modelscope_download(args.qwen_model, local, args.dry_run)
                task["status"] = "ok"
                task["code_dir"] = str(dirs["qwen_code"] / "Qwen2.5-VL")
                task["weights_dir"] = str(local)
            except Exception as exc:
                task["status"] = "failed"
                task["error"] = repr(exc)
                tqdm.write(f"QwenVL failed: {exc}")
            manifest["tasks"]["qwen"] = task
            bar.update(1)

        if not args.skip_locate:
            task = {"status": "pending", "model": args.locate_model}
            try:
                task["code"] = try_clone_repo(
                    "https://github.com/NVlabs/Eagle.git", dirs["locate_code"] / "Eagle", args.dry_run
                )
                local = dirs["locate_weights"] / args.locate_model.split("/")[-1]
                try:
                    snapshot_download(args.locate_model, local, args.dry_run)
                except Exception as hf_exc:
                    tqdm.write(f"LocateAnything HF download failed: {hf_exc}")
                    try_modelscope_download("nv-community/LocateAnything-3B", local, args.dry_run)
                task["status"] = "ok"
                task["code_dir"] = str(dirs["locate_code"] / "Eagle")
                task["weights_dir"] = str(local)
            except Exception as exc:
                task["status"] = "failed"
                task["error"] = repr(exc)
                tqdm.write(f"LocateAnything failed: {exc}")
            manifest["tasks"]["locate"] = task
            bar.update(1)

    out = write_manifest(dirs["results"], manifest)
    tqdm.write(f"Manifest written to: {out}")

    failed = [name for name, task in manifest["tasks"].items() if isinstance(task, dict) and task.get("status") == "failed"]
    if failed:
        tqdm.write("Some downloads failed: " + ", ".join(failed))
        tqdm.write("You can rerun this script; completed folders are skipped and HF downloads resume.")
        return 2

    tqdm.write("All requested code/weight downloads completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
