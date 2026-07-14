"""Benchmark the eight models used in the vegetation-segmentation study.

This is an inference-only script: it loads the selected existing checkpoints and
does not modify any model, dataset, or manuscript file.  Timings are GPU forward
pass timings (preprocessing, disk I/O, visualization, and CPU postprocessing are
excluded).  Native training resolutions are intentionally retained and recorded
in the output CSV; SAM-family models were trained with a 1024-pixel input.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import os
import pkgutil
import sys
import time
import traceback
import types
import zipimport
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import torch
import torch.nn as nn

# Python 3.12 removed pkgutil.ImpImporter. Older pkg_resources code imported by
# some Ultralytics/THOP dependency combinations still accesses it at import time.
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = zipimport.zipimporter  # type: ignore[attr-defined]


def looks_like_project_root(path: Path) -> bool:
    return (path / "1_SAM2_Tiny").is_dir() and (path / "8_DeepLabV3").is_dir()


def discover_project_root() -> Path:
    """Work both from <project>/scripts and from a downloaded /root/scripts copy."""
    script_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    candidates = [
        Path(os.environ["VEGETATION_PROJECT_ROOT"]).expanduser()
        if os.environ.get("VEGETATION_PROJECT_ROOT")
        else script_root,
        script_root,
        cwd,
        cwd / "vegetation_models_v2",
        Path("/root/vegetation_models_v2"),
        Path("/root/autodl-tmp/vegetation_models_v2"),
    ]
    for candidate in candidates:
        if looks_like_project_root(candidate):
            return candidate.resolve()
    return script_root


ROOT = discover_project_root()
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "model_efficiency_stats.csv"
REQUIRE_FINETUNED = False


def configure_project_root(path: str) -> None:
    """Set a user-supplied project root after argument parsing."""
    global ROOT, RESULTS_DIR, CSV_PATH
    candidate = Path(path).expanduser().resolve()
    if not looks_like_project_root(candidate):
        raise ValueError(
            f"{candidate} is not the project root; expected directories such as "
            "1_SAM2_Tiny and 8_DeepLabV3."
        )
    ROOT = candidate
    RESULTS_DIR = ROOT / "results"
    CSV_PATH = RESULTS_DIR / "model_efficiency_stats.csv"

FIELDS = [
    "Model",
    "Params_M",
    "FLOPs_G",
    "Inference_Time_ms",
    "FPS",
    "GPU_Memory_MB",
    "Input_Size",
    "Device",
    "Notes",
]


class EncoderSegmentationModel(nn.Module):
    """The semantic head used by the SAM2/MobileSAM training notebooks."""

    def __init__(self, encoder: nn.Module, output_size: int = 1024):
        super().__init__()
        self.encoder = encoder
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
            nn.Upsample(size=(output_size, output_size), mode="bilinear", align_corners=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        # SAM2 returns a feature dictionary, while MobileSAM returns a tensor.
        if isinstance(features, dict):
            features = features["backbone_fpn"][-1]
        return self.seg_head(features)


def add_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def purge_module_tree(package: str) -> None:
    """Drop already-imported modules so local project copies take precedence."""
    for name in list(sys.modules):
        if name == package or name.startswith(f"{package}."):
            del sys.modules[name]


def get_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Read the notebook-style checkpoint without accepting a random fallback."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint):
        return checkpoint
    raise RuntimeError(f"No model state dictionary found in {path}")


def load_finetuned_if_available(model: nn.Module, checkpoint: Path) -> str:
    """Load the experiment checkpoint, or state explicitly why it was unavailable."""
    if not checkpoint.is_file():
        if REQUIRE_FINETUNED:
            raise FileNotFoundError(f"Fine-tuned checkpoint absent: {checkpoint}")
        return f"Fine-tuned checkpoint absent ({checkpoint.name}); architecture-only measurement."
    if checkpoint.stat().st_size == 0:
        if REQUIRE_FINETUNED:
            raise FileNotFoundError(f"Fine-tuned checkpoint is empty: {checkpoint}")
        return f"Fine-tuned checkpoint is empty ({checkpoint.name}); architecture-only measurement."
    model.load_state_dict(get_state_dict(checkpoint))
    return f"Fine-tuned {checkpoint.name}."


def load_unet(device: torch.device) -> Tuple[nn.Module, int, str]:
    add_path(ROOT / "7_UNet" / "code")
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=2, activation=None
    )
    checkpoint = ROOT / "7_UNet" / "checkpoints" / "unet_best.pth"
    status = load_finetuned_if_available(model, checkpoint)
    return model.to(device).eval(), 320, f"{status} Native training resolution."


def load_deeplab(device: torch.device) -> Tuple[nn.Module, int, str]:
    add_path(ROOT / "7_UNet" / "code")
    import segmentation_models_pytorch as smp

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=2,
        encoder_output_stride=16,
        activation=None,
    )
    checkpoint = ROOT / "8_DeepLabV3" / "checkpoints" / "deeplabv3plus_best.pth"
    status = load_finetuned_if_available(model, checkpoint)
    return model.to(device).eval(), 512, f"{status} Native training resolution."


def load_mobilesam(device: torch.device, version: str) -> Tuple[nn.Module, int, str]:
    directory = "2_MobileSAM" if version == "MobileSAM" else "6_MobileSAMV2"
    add_path(ROOT / directory / "code")
    # Both project notebooks use the TinyViT MobileSAM image encoder for semantic masks.
    from mobile_sam import sam_model_registry

    base = ROOT / directory / "weights" / "mobile_sam.pt"
    fine = ROOT / directory / "checkpoints" / ("mobilesam_best.pth" if version == "MobileSAM" else "mobilesamv2_best.pth")
    sam = sam_model_registry["vit_t"](checkpoint=str(base))
    model = EncoderSegmentationModel(sam.image_encoder)
    status = load_finetuned_if_available(model, fine)
    return model.to(device).eval(), 1024, f"{status} TinyViT encoder and semantic head as in the training notebook."


def load_sam2(device: torch.device, version: str) -> Tuple[nn.Module, int, str]:
    add_path(ROOT / "1_SAM2_Tiny" / "code")
    purge_module_tree("sam2")
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2

    if version == "SAM2-Tiny":
        config_name = "sam2/sam2_hiera_t.yaml"
        base = ROOT / "1_SAM2_Tiny" / "weights" / "sam2_hiera_tiny.pt"
        fine = ROOT / "1_SAM2_Tiny" / "checkpoints" / "sam2tiny_best.pth"
    else:
        config_name = "sam2.1/sam2.1_hiera_t.yaml"
        base = ROOT / "5_SAM21_Tiny" / "weights" / "sam2.1_hiera_tiny.pt"
        fine = ROOT / "5_SAM21_Tiny" / "checkpoints" / "sam21tiny_best.pth"

    config_dir = ROOT / "1_SAM2_Tiny" / "code" / "sam2" / "configs"
    # A single process benchmarks SAM2 and SAM2.1 sequentially. Hydra normally
    # cleans this context, but some releases leave global state initialized.
    global_hydra = GlobalHydra.instance()
    if global_hydra.is_initialized():
        global_hydra.clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        sam = build_sam2(config_name, ckpt_path=str(base), device=device, mode="eval")
    if global_hydra.is_initialized():
        global_hydra.clear()
    model = EncoderSegmentationModel(sam.image_encoder)
    status = load_finetuned_if_available(model, fine)
    return model.to(device).eval(), 1024, f"{status} Hiera-tiny encoder and semantic head as in the training notebook."


def load_yolo(device: torch.device) -> Tuple[nn.Module, int, str]:
    from ultralytics import YOLO

    checkpoint = ROOT / "3_YOLO11seg" / "checkpoints" / "yolo11s_vegetation" / "weights" / "best.pt"
    yolo = YOLO(str(checkpoint))
    # Raw forward measures GPU network cost only, with no CPU image decode or NMS.
    return yolo.model.to(device).eval(), 512, "Fine-tuned best.pt; raw network forward only (excludes Ultralytics preprocessing/NMS)."


def install_segearth_lightweight_stubs() -> None:
    """Avoid importing OpenMMLab CUDA extensions that SegEarth does not use here."""

    class BaseSegmentor(nn.Module):
        def __init__(self, data_preprocessor=None, *args, **kwargs):
            super().__init__()
            self.data_preprocessor = data_preprocessor

    class SegDataPreProcessor:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class PixelData:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Registry:
        def register_module(self, *args, **kwargs):
            def decorator(cls):
                return cls
            return decorator

    mmseg = types.ModuleType("mmseg")
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_segmentors = types.ModuleType("mmseg.models.segmentors")
    mmseg_data_preprocessor = types.ModuleType("mmseg.models.data_preprocessor")
    mmseg_registry = types.ModuleType("mmseg.registry")
    mmengine = types.ModuleType("mmengine")
    mmengine_structures = types.ModuleType("mmengine.structures")

    mmseg_segmentors.BaseSegmentor = BaseSegmentor
    mmseg_data_preprocessor.SegDataPreProcessor = SegDataPreProcessor
    mmseg_registry.MODELS = _Registry()
    mmengine_structures.PixelData = PixelData

    sys.modules["mmseg"] = mmseg
    sys.modules["mmseg.models"] = mmseg_models
    sys.modules["mmseg.models.segmentors"] = mmseg_segmentors
    sys.modules["mmseg.models.data_preprocessor"] = mmseg_data_preprocessor
    sys.modules["mmseg.registry"] = mmseg_registry
    sys.modules["mmengine"] = mmengine
    sys.modules["mmengine.structures"] = mmengine_structures


def patch_transformers_for_blip_import() -> None:
    """Support SegEarth's vendored BLIP import under newer transformers."""
    try:
        import transformers.modeling_utils as modeling_utils
        if hasattr(modeling_utils, "apply_chunking_to_forward"):
            return
        from transformers.pytorch_utils import apply_chunking_to_forward
        modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    except Exception:
        # If transformers is absent or has a different layout, let the original
        # import error surface with its full message.
        return


def load_segearth(device: torch.device) -> Tuple[nn.Module, int, str]:
    """Load the uploaded CLIP checkpoint locally, without a network download."""
    code = ROOT / "4_SegEarth_OV" / "code"
    add_path(code)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    install_segearth_lightweight_stubs()
    patch_transformers_for_blip_import()
    # SegEarth's constructor requests pretrained='openai'. Its vendored OpenCLIP accepts
    # a local checkpoint path, so redirect only that request to the uploaded weight.
    segearth = importlib.import_module("segearth_segmentor")
    original_create_model = segearth.create_model
    local_clip = ROOT / "4_SegEarth_OV" / "weights" / "pytorch_model.bin"

    def local_create_model(name: str, pretrained=None, *args, **kwargs):
        if pretrained == "openai":
            pretrained = str(local_clip)
        return original_create_model(name, pretrained=pretrained, *args, **kwargs)

    segearth.create_model = local_create_model
    class_file = RESULTS_DIR / "segearth_efficiency_classes.txt"
    class_file.write_text("background\nvegetation\n", encoding="utf-8")
    model = segearth.SegEarthSegmentation(
        clip_type="CLIP",
        vit_type="ViT-B/16",
        model_type="SegEarth",
        ignore_residual=True,
        feature_up=True,
        feature_up_cfg={
            "model_name": "jbu_one",
            "model_path": str(code / "simfeatup_dev" / "weights" / "xclip_jbu_one_million_aid.ckpt"),
        },
        cls_token_lambda=-0.3,
        name_path=str(class_file),
        prob_thd=0.1,
        device=device,
    )

    class SegEarthForward(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x):
            return self.wrapped.forward_feature(x)

    return SegEarthForward(model).to(device).eval(), 448, "CLIP ViT-B/16 + SimFeatUp; forward_feature only (text prompts precomputed, no sliding-window aggregation)."


LOADERS: Dict[str, Callable[[torch.device], Tuple[nn.Module, int, str]]] = {
    "SAM2-Tiny": lambda d: load_sam2(d, "SAM2-Tiny"),
    "SAM2.1-Tiny": lambda d: load_sam2(d, "SAM2.1-Tiny"),
    "MobileSAM": lambda d: load_mobilesam(d, "MobileSAM"),
    "MobileSAMV2": lambda d: load_mobilesam(d, "MobileSAMV2"),
    "UNet": load_unet,
    "DeepLabV3+": load_deeplab,
    "YOLO11s-seg": load_yolo,
    "SegEarth-OV": load_segearth,
}


def model_flops(model: nn.Module, sample: torch.Tensor) -> Tuple[str, str]:
    """Return MAC-derived FLOPs when THOP can trace the model, otherwise N/A."""
    try:
        # Python 3.12 removed stdlib distutils. Some THOP/timm combinations still
        # import it, while setuptools provides a compatible maintained copy.
        try:
            import distutils  # noqa: F401
        except ModuleNotFoundError:
            import setuptools._distutils as setuptools_distutils

            sys.modules["distutils"] = setuptools_distutils
        from thop import profile

        with torch.inference_mode():
            macs, _ = profile(model, inputs=(sample,), verbose=False)
        # A multiply-accumulate represents two floating-point operations.
        return f"{2.0 * macs / 1e9:.3f}", "FLOPs=2×THOP MACs for one forward pass."
    except Exception as thop_exc:
        try:
            # Fallback for models that THOP cannot safely hook, especially
            # attention-heavy SAM2 blocks under newer Python/PyTorch stacks.
            with torch.profiler.profile(with_flops=True) as prof:
                with torch.inference_mode():
                    _ = model(sample)
            flops = sum(event.flops for event in prof.key_averages() if event.flops)
            if flops > 0:
                return (
                    f"{flops / 1e9:.3f}",
                    "FLOPs=PyTorch profiler fallback for one forward pass "
                    f"(THOP unavailable: {type(thop_exc).__name__}).",
                )
        except Exception:
            pass
        return "N/A", f"FLOPs unavailable: {type(thop_exc).__name__}: {str(thop_exc).splitlines()[0][:180]}"


def benchmark_forward(model: nn.Module, sample: torch.Tensor, warmup: int, repeats: int) -> Tuple[float, float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(sample)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(sample.device)
        started = time.perf_counter()
        for _ in range(repeats):
            _ = model(sample)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    ms = elapsed * 1000.0 / repeats
    fps = 1000.0 / ms
    peak_mb = torch.cuda.max_memory_allocated(sample.device) / (1024.0 * 1024.0)
    return ms, fps, peak_mb


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def failed_row(name: str, device: str, exc: BaseException) -> Dict[str, str]:
    message = " ".join(traceback.format_exception_only(type(exc), exc)).strip().replace("\n", " ")
    return {
        "Model": name,
        "Params_M": "N/A",
        "FLOPs_G": "N/A",
        "Inference_Time_ms": "N/A",
        "FPS": "N/A",
        "GPU_Memory_MB": "N/A",
        "Input_Size": "N/A",
        "Device": device,
        "Notes": f"NOT MEASURED — {message[:500]}",
    }


def run_one(name: str, device: torch.device, warmup: int, repeats: int, compute_flops: bool) -> Dict[str, str]:
    empty_cuda()
    model = None
    try:
        model, size, notes = LOADERS[name](device)
        # SegEarth's official implementation runs its CLIP/upsampler in FP16;
        # the other models are FP32. Match the loaded model rather than silently
        # casting it, so the recorded precision reflects its real inference path.
        parameter = next(model.parameters())
        sample = torch.randn(1, 3, size, size, device=device, dtype=parameter.dtype)
        params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
        # Measure latency before FLOPs. Third-party FLOPs tracers may attach hooks
        # to unsupported modules; a FLOPs failure must not invalidate timing.
        torch.cuda.synchronize()
        ms, fps, peak_mb = benchmark_forward(model, sample, warmup, repeats)
        flops, flops_note = model_flops(model, sample) if compute_flops else ("N/A", "FLOPs disabled by --no-flops.")
        return {
            "Model": name,
            "Params_M": f"{params_m:.3f}",
            "FLOPs_G": flops,
            "Inference_Time_ms": f"{ms:.3f}",
            "FPS": f"{fps:.2f}",
            "GPU_Memory_MB": f"{peak_mb:.1f}",
            "Input_Size": f"{size}x{size}",
            "Device": torch.cuda.get_device_name(device),
            "Notes": f"{notes} {flops_note} Warm-up={warmup}, timed repeats={repeats}, batch=1, FP32.",
        }
    except Exception as exc:
        return failed_row(name, str(device), exc)
    finally:
        del model
        empty_cuda()


def write_csv(rows: Iterable[Dict[str, str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def merge_rows(existing_rows: Iterable[Dict[str, str]], new_rows: Iterable[Dict[str, str]], target_models: set[str]) -> list[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for row in existing_rows:
        model = row.get("Model", "")
        if model and model not in target_models:
            merged[model] = {field: row.get(field, "") for field in FIELDS}
    for row in new_rows:
        model = row["Model"]
        merged[model] = {field: row.get(field, "") for field in FIELDS}
    order = {name: index for index, name in enumerate(LOADERS)}
    return sorted(merged.values(), key=lambda row: order.get(row["Model"], len(order)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-only efficiency benchmark for the eight study models.")
    parser.add_argument(
        "--project-root",
        default=str(ROOT),
        help="Absolute project directory. Useful when this script was downloaded outside the project tree.",
    )
    parser.add_argument("--models", nargs="*", choices=list(LOADERS), default=list(LOADERS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--no-flops", action="store_true", help="Skip THOP tracing; use this for a fast timing-only dry run.")
    parser.add_argument("--require-finetuned", action="store_true", help="Mark models as NOT MEASURED when their fine-tuned checkpoints are absent or empty.")
    args = parser.parse_args()

    try:
        configure_project_root(args.project_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the requested GPU latency benchmark.")
    if args.warmup < 1 or args.repeats < 1:
        raise SystemExit("--warmup and --repeats must both be positive.")

    global REQUIRE_FINETUNED
    REQUIRE_FINETUNED = args.require_finetuned

    device = torch.device("cuda:0")
    print(f"Repository: {ROOT}")
    print(f"GPU: {torch.cuda.get_device_name(device)} | PyTorch: {torch.__version__}")
    print(f"Protocol: batch=1, FP32, native input size, warm-up={args.warmup}, repeats={args.repeats}")
    existing_rows = read_csv(CSV_PATH)
    target_models = set(args.models)
    rows = []
    for name in args.models:
        print(f"\n[{name}] loading and benchmarking...")
        row = run_one(name, device, args.warmup, args.repeats, not args.no_flops)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        write_csv(merge_rows(existing_rows, rows, target_models))  # Preserve completed rows if a later third-party model fails.
    print(f"\nSaved: {CSV_PATH}")


if __name__ == "__main__":
    main()
