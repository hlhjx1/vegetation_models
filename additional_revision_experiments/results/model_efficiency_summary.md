# Model Efficiency Statistics

Reviewer comment addressed: "What are the parameters and inference time of all models?"

Benchmark protocol: batch size 1; 10 warm-up iterations; 100 timed forward passes; CUDA synchronization before and after timing; reported memory is peak GPU memory during the measured forward pass. Timings exclude image loading, visualization, and CPU post-processing. FLOPs are reported as 2 x MACs when THOP is available; otherwise the PyTorch profiler fallback is noted.

## Final Table

| Model | Params (M) | FLOPs (G) | Inference time (ms/img) | FPS | GPU memory (MB) | Input size | Device | Notes |
|---|---:|---:|---:|---:|---:|---|---|---|
| SAM2-Tiny | 28.179 | 213.325 | 11.455 | 87.30 | 501.4 | 1024x1024 | NVIDIA RTX PRO 6000 Blackwell Server Edition | Fine-tuned `sam2tiny_best.pth`; FLOPs from PyTorch profiler fallback because THOP was unavailable for this model. |
| SAM2.1-Tiny | 28.179 | 213.769 | 11.414 | 87.61 | 503.3 | 1024x1024 | NVIDIA RTX PRO 6000 Blackwell Server Edition | `sam21tiny_best.pth` is empty; architecture-only measurement using SAM2.1-Tiny weights and semantic head. |
| MobileSAM | 7.025 | 81.399 | 6.534 | 153.04 | 278.5 | 1024x1024 | NVIDIA RTX PRO 6000 Blackwell Server Edition | Fine-tuned `mobilesam_best.pth`; TinyViT encoder and semantic head as in the training notebook. |
| MobileSAMV2 | 7.025 | 81.399 | 6.552 | 152.62 | 278.5 | 1024x1024 | NVIDIA RTX PRO 6000 Blackwell Server Edition | Fine-tuned `mobilesamv2_best.pth`; TinyViT encoder and semantic head as in the training notebook. |
| UNet | 24.437 | 24.560 | 2.399 | 416.79 | 154.0 | 320x320 | NVIDIA RTX PRO 6000 Blackwell Server Edition | Fine-tuned `unet_best.pth`; native training resolution. |
| DeepLabV3+ | 26.678 | 73.822 | 2.970 | 336.66 | 242.3 | 512x512 | NVIDIA RTX PRO 6000 Blackwell Server Edition | Fine-tuned `deeplabv3plus_best.pth`; native training resolution. |
| YOLO11s-seg | 10.083 | 22.779 | 5.251 | 190.44 | 97.4 | 512x512 | NVIDIA RTX PRO 6000 Blackwell Server Edition | Fine-tuned `best.pt`; raw network forward only, excluding Ultralytics preprocessing and NMS. |
| SegEarth-OV | 149.914 | 211.451 | 36.195 | 27.63 | 3531.7 | 448x448 | NVIDIA RTX PRO 6000 Blackwell Server Edition | Colab run; CLIP ViT-B/16 + SimFeatUp JBU; `forward_feature` only; batch 1, FP16; FLOPs from THOP. |

## Short Summary

UNet achieved the highest FPS among the tested models, followed by DeepLabV3+. YOLO11s-seg had the lowest parameter count and GPU memory footprint among the non-SAM semantic/instance segmentation baselines. SegEarth-OV had the largest parameter count and memory footprint, mainly because it uses CLIP ViT-B/16 with the SimFeatUp JBU module. SAM2-Tiny and SAM2.1-Tiny had similar computational costs because they share the Hiera-tiny architecture.

## Suggested Manuscript Text

To address the reviewers' concern regarding computational efficiency, we measured the number of parameters, FLOPs, single-image inference latency, FPS, and peak GPU memory usage for all compared models. All latency results were obtained after 10 warm-up iterations and averaged over 100 forward passes with CUDA synchronization. The results show that UNet and DeepLabV3+ provide the fastest inference, while YOLO11s-seg has the lowest memory footprint. In contrast, SegEarth-OV requires substantially higher memory and inference time due to its CLIP ViT-B/16 backbone and SimFeatUp feature upsampling module.

## Suggested Response Draft

Thank you for this valuable suggestion. We have added a computational-cost comparison for all evaluated models, including the number of parameters, FLOPs, single-image inference time, FPS, and peak GPU memory usage. The inference time was measured after 10 warm-up iterations and averaged over 100 forward passes with CUDA synchronization. The added results show the efficiency trade-offs among the compared models: UNet and DeepLabV3+ achieve the highest inference speed, YOLO11s-seg has the smallest memory footprint, and SegEarth-OV has the highest computational cost because of its CLIP ViT-B/16 backbone and SimFeatUp module.

## Script References

- `scripts/benchmark_model_efficiency.py`
- `scripts/install_efficiency_deps_autodl.sh`

## Supplement: Current Four Foundation/Prompt Models

This section is appended for the additional first-round revision experiments. It does not modify the old eight-model efficiency table above. The four models below have different task modes, so only DINOv2 and DINOv3 are directly comparable as frozen-backbone segmentation models. QwenVL2.5-3B and LocateAnything are prompt-based inference or grounding supplements; their box-mask outputs should be reported as additional evidence rather than as dense segmentation training baselines.

### Available Result Sources Checked

- DINOv2 four-dataset results: `results/dinov2_four_dataset_results.csv`
- DINOv3 four-dataset results: `results/dinov3_four_dataset_results.csv`
- DINOv2 trained heads: `9_DINOv2/four_dataset_runs/*/best_head.pth`
- DINOv3 trained heads: `10_DINOv3/four_dataset_runs/*/best_head.pth`
- QwenVL raw and CSV outputs: `11_QwenVL/prompt_eval/qwenvl_raw_outputs.jsonl`, `results/qwenvl_prompt_eval_results.csv`
- LocateAnything raw and CSV outputs: `12_LocatingAnything/grounding_eval/locateanything_raw_outputs.jsonl`, `results/locateanything_grounding_results.csv`
- LocateAnything/QwenVL summary: `results/locateanything_grounding_summary.md`

### Four-Model Efficiency Status

| Model | Task type | Existing performance source | Parameters | FLOPs/GFLOPs | Inference time/FPS | Current status |
|---|---|---|---|---|---|---|
| DINOv2 ViT-L/14 + frozen segmentation head | frozen-head segmentation | `results/dinov2_four_dataset_results.csv`; heads under `9_DINOv2/four_dataset_runs/*/best_head.pth` | Backbone params pending exact Colab count; trained head counted locally as 0.558466 M params per dataset head | Pending; run Colab measurement below with backbone + dataset-specific head | Pending; run Colab measurement below, batch=1, warm-up + repeated CUDA timing | Needs Colab efficiency run because the local result file records training elapsed time, not single-image inference latency |
| DINOv3 ViT-L/16 SAT-493M + frozen segmentation head | frozen-head segmentation | `results/dinov3_four_dataset_results.csv`; heads under `10_DINOv3/four_dataset_runs/*/best_head.pth` | Backbone params pending exact Colab count; trained head counted locally as 0.558466 M params per dataset head | Pending; run Colab measurement below with local DINOv3 repo and weights | Pending; run Colab measurement below, batch=1, warm-up + repeated CUDA timing | Needs Colab efficiency run because the local result file records training elapsed time, not single-image inference latency |
| Qwen/Qwen2.5-VL-3B-Instruct | prompt-based inference | `results/qwenvl_prompt_eval_results.csv` | About 3B from model name; exact loaded parameter count pending Colab check | Not yet measured; conventional FLOPs are not directly comparable to dense segmentation forward FLOPs because generation length and prompt/image tokens affect cost | Not yet measured; should be measured as prompt-to-box end-to-end latency with fixed prompt and max_new_tokens | Supplemental prompt-based box-mask inference only |
| nvidia/LocateAnything-3B | grounding | `results/locateanything_grounding_results.csv` | About 3B from model name; exact loaded parameter count pending Colab check | Not yet measured; grounding/generation cost is prompt and output dependent | Not yet measured; should be measured as prompt-to-box end-to-end latency with fixed prompt and decoding settings | Supplemental grounding box-mask inference only |

### DINOv2/DINOv3 Training-Result Context

These values are already present in the current result CSV files and are not latency/FLOPs measurements.

| Model | Dataset | Eval samples | Input size | Best mIoU | Veg IoU | F1 | Accuracy | Training/eval elapsed seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 ViT-L/14 reg | Zijinshan | 67 | 518 | 0.7466 | 0.8161 | 0.8987 | 0.8673 | 160.73 |
| DINOv2 ViT-L/14 reg | LoveDA | 100 | 518 | 0.7076 | 0.4839 | 0.6522 | 0.9355 | 358.68 |
| DINOv2 ViT-L/14 reg | Potsdam | 80 | 518 | 0.8403 | 0.8633 | 0.9267 | 0.9152 | 277.13 |
| DINOv2 ViT-L/14 reg | Vaihingen | 40 | 518 | 0.8782 | 0.8616 | 0.9256 | 0.9364 | 147.40 |
| DINOv3 ViT-L/16 SAT-493M | Zijinshan | 67 | 512 | 0.7683 | 0.8379 | 0.9118 | 0.8822 | 76.03 |
| DINOv3 ViT-L/16 SAT-493M | LoveDA | 100 | 512 | 0.7505 | 0.5621 | 0.7197 | 0.9433 | 332.74 |
| DINOv3 ViT-L/16 SAT-493M | Potsdam | 80 | 512 | 0.8530 | 0.8757 | 0.9337 | 0.9227 | 271.76 |
| DINOv3 ViT-L/16 SAT-493M | Vaihingen | 40 | 512 | 0.8793 | 0.8619 | 0.9258 | 0.9372 | 159.33 |

### Prompt/Grounding Box-Mask Results Already Available

QwenVL2.5-3B and LocateAnything are not dense segmentation training methods. Their boxes were converted to coarse binary masks only for supplemental box-mask IoU/F1/accuracy.

| Model | Dataset | n | Box-mask IoU | Box-mask F1 | Box-mask Acc | Avg boxes |
|---|---|---:|---:|---:|---:|---:|
| QwenVL2.5-3B | Zijinshan | 100 | 0.6212 | 0.6981 | 0.7557 | 1.15 |
| QwenVL2.5-3B | LoveDA | 100 | 0.1175 | 0.1806 | 0.6437 | 1.09 |
| QwenVL2.5-3B | Potsdam | 100 | 0.3482 | 0.4233 | 0.6145 | 0.73 |
| QwenVL2.5-3B | Vaihingen | 100 | 0.4615 | 0.6080 | 0.5290 | 1.11 |
| QwenVL2.5-3B | ALL | 400 | 0.3871 | 0.4775 | 0.6357 | 1.02 |
| LocateAnything-3B | LoveDA | 100 | 0.1122 | 0.1756 | 0.6649 | 16.89 |
| LocateAnything-3B | Potsdam | 100 | 0.5038 | 0.5914 | 0.6932 | 2.74 |
| LocateAnything-3B | Vaihingen | 100 | 0.5444 | 0.6924 | 0.7046 | 8.85 |
| LocateAnything-3B | Zijinshan | 100 | 0.5747 | 0.6457 | 0.7055 | 13.64 |
| LocateAnything-3B | ALL | 400 | 0.4338 | 0.5263 | 0.6920 | 10.53 |

### Fair-Comparison Notes

- Fair dense/frozen-head comparison: DINOv2 vs DINOv3 can be compared with the segmentation-head models because both use frozen visual backbones plus trained segmentation heads. FLOPs and latency should include both the frozen backbone and the segmentation head.
- Supplemental comparison only: QwenVL2.5-3B and LocateAnything should not be mixed into the old dense segmentation efficiency ranking. They perform prompt-based localization/grounding, and their latency depends on prompt tokens, image tokens, generated tokens, remote-code implementation, and decoding settings.
- Existing old-model benchmark values above remain valid for Reviewer 1 Comment 5 and Reviewer 2 Comment 6. The four-model supplement should be presented as an added foundation/prompt-model analysis once the pending Colab latency/FLOPs run is completed.

### Colab Script for Locating Code/Weights and Measuring Pending Items

The Colab `/content` environment should contain the four model code/weight folders. DINOv2 and DINOv3 also need the local best segmentation-head weights uploaded or synced to the same relative paths as this project, for example `9_DINOv2/four_dataset_runs/zijinshan/best_head.pth` and `10_DINOv3/four_dataset_runs/zijinshan/best_head.pth`.

```python
from pathlib import Path
import json
import time
import torch

PROJECT = Path("/content/vegetation_models_v2")
CANDIDATE_ROOTS = [
    PROJECT,
    Path("/content/drive/MyDrive/vegetation_models_v2"),
    Path("/content/vegetation_foundation_models"),
    Path("/root/autodl-tmp/vegetation_foundation_models"),
]

def existing(paths):
    return [str(p) for p in paths if p.exists()]

paths = {
    "DINOv2_code_or_cache": [
        PROJECT / "9_DINOv2",
        Path("/root/.cache/torch/hub/facebookresearch_dinov2_main"),
        Path("/content/vegetation_foundation_models/_cache/torch_cache/hub/facebookresearch_dinov2_main"),
    ],
    "DINOv2_pretrain": [
        Path("/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"),
        PROJECT / "_cache/torch_cache/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth",
        Path("/content/vegetation_foundation_models/_cache/torch_cache/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"),
    ],
    "DINOv2_heads": [PROJECT / "9_DINOv2/four_dataset_runs" / d / "best_head.pth" for d in ["zijinshan", "loveda", "potsdam", "vaihingen"]],
    "DINOv3_repo": [
        PROJECT / "10_DINOv3/code/dinov3-main",
        Path("/content/drive/MyDrive/vegetation_models_v2/10_DINOv3/code/dinov3-main"),
    ],
    "DINOv3_pretrain": [
        PROJECT / "10_DINOv3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        Path("/content/drive/MyDrive/vegetation_models_v2/10_DINOv3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
    ],
    "DINOv3_heads": [PROJECT / "10_DINOv3/four_dataset_runs" / d / "best_head.pth" for d in ["zijinshan", "loveda", "potsdam", "vaihingen"]],
    "QwenVL_weights_or_cache": [
        PROJECT / "11_QwenVL/weights",
        Path("/content/drive/MyDrive/vegetation_models_v2/11_QwenVL/weights"),
        Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"),
    ],
    "LocateAnything_weights_or_cache": [
        PROJECT / "12_LocatingAnything/weights",
        Path("/content/drive/MyDrive/vegetation_models_v2/12_LocatingAnything/weights"),
        Path("/root/.cache/huggingface/hub/models--nvidia--LocateAnything-3B"),
    ],
}

print(json.dumps({k: {"found": existing(v), "checked": [str(p) for p in v]} for k, v in paths.items()}, indent=2))

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_forward(model, sample, warmup=10, repeats=100):
    model.eval().cuda()
    sample = sample.cuda()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(sample)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000.0 / repeats
    return ms, 1000.0 / ms

# DINOv2/DINOv3 measurement plan:
# 1. Import or copy the DINO segmentation model classes from scripts/run_dinov2_four_dataset_train.py
#    and scripts/run_dinov3_four_dataset_train.py.
# 2. Load the frozen backbone and the dataset-specific best_head.pth.
# 3. Run benchmark_forward on a dummy image tensor of shape [1, 3, 518, 518] for DINOv2
#    and [1, 3, 512, 512] for DINOv3.
# 4. Use thop.profile(model, inputs=(sample,)) when THOP supports the modules; otherwise report
#    FLOPs as pending/profiler fallback.

# QwenVL/LocateAnything latency plan:
# Measure end-to-end prompt-to-box latency on the same fixed 100 samples per dataset with fixed
# prompt, fixed max_new_tokens, and torch.cuda.synchronize around model.generate().
# Do not compare these latency numbers directly with dense segmentation forward-only latency.
```

### Reviewer-Response Points for the Four-Model Supplement

For Reviewer 1 Comment 5 and Reviewer 2 Comment 6, the current response can state that computational costs for the original dense segmentation models were measured in the main efficiency table, while the newly added DINOv2/DINOv3 frozen-head models require an additional Colab benchmark including the local best heads. QwenVL2.5-3B and LocateAnything-3B are prompt-based inference/grounding models, so their parameter counts can be reported from the loaded models, but FLOPs and latency should be disclosed as prompt/generation-dependent and kept separate from dense segmentation forward-pass comparisons.
