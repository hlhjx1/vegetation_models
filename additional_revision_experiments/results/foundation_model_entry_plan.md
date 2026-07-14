# Added Foundation-Model Experiment Entry Plan

This project stage should use the current 2024 vegetation segmentation dataset first:
`datasets/2024-seg/JPEGImages`, `datasets/2024-seg/SegmentationClass`, and
`datasets/2024-seg/ImageSets/Segmentation/default.txt`.

## Dataset protocol

- Smoke tests: use only the Zijinshan/2024 dataset currently in this project.
- Formal training: do not switch to four-dataset training without an explicit protocol decision.
- Four-dataset experiments are cross-dataset/generalization experiments, not the default replacement for the current revision-stage main dataset.

## DINOv2

- Role: standard dense semantic segmentation add-on model.
- Entry: `scripts/run_dinov2_smoke_train.py`.
- Model: frozen `dinov2_vitl14_reg` backbone plus a small convolutional segmentation head.
- Output: `9_DINOv2/smoke_outputs/` and `results/dinov2_smoke_train_log.csv`.
- Current local status: missing local torch hub repo/weight cache. Colab path reported by user is ready:
  `/content/vegetation_foundation_models/_cache/torch_cache/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`.

## DINOv3

- Role: reserved dense semantic segmentation add-on model after gated weights are available.
- Current action: path/status detection only.
- Entry rule: if `10_DINOv3/weights` or the AutoDL/Colab DINOv3 weight directory has no files, skip with a clear gated-access message.
- Do not make DINOv3 the first formal training target until HF access is solved.

## QwenVL

- Role: prompt-based VLM supplement, not a standard pixel-level semantic segmentation model.
- Planned smoke test: load `Qwen/Qwen2.5-VL-3B-Instruct`, run 1-3 sample images with prompts asking for vegetation presence and rough location, save JSON/text responses under `11_QwenVL/`.
- If masks are needed, use an explicitly separate protocol such as converting text/boxes to rough masks. Do not interpret this as equivalent to dense semantic segmentation.

## LocateAnything / LocatingAnything

- Role: prompt-based grounding/location supplement.
- Planned smoke test: load `nvidia/LocateAnything-3B`, prompt for vegetation/forest regions, save predicted boxes/regions under `12_LocatingAnything/`.
- Optional metrics may convert boxes to binary box masks, but the metric should be labeled as a grounding-derived supplemental metric.

## Copyable commands

Status check:

```bash
python scripts/check_foundation_model_status.py \
  --project-root /content/vegetation_models_v2 \
  --storage-root /content/vegetation_foundation_models \
  --write-json results/foundation_model_status.json
```

DINOv2 smoke train on Colab:

```bash
cd /content/vegetation_models_v2
python scripts/run_dinov2_smoke_train.py \
  --project-root /content/vegetation_models_v2 \
  --torch-cache /content/vegetation_foundation_models/_cache/torch_cache \
  --epochs 1 \
  --batch-size 1 \
  --input-size 518 \
  --max-train-samples 12 \
  --max-val-samples 8
```

DINOv2 smoke train on AutoDL:

```bash
cd /root/vegetation_models_v2
python scripts/run_dinov2_smoke_train.py \
  --project-root /root/vegetation_models_v2 \
  --torch-cache /root/autodl-tmp/vegetation_foundation_models/_cache/torch_cache \
  --epochs 1 \
  --batch-size 1 \
  --input-size 518 \
  --max-train-samples 12 \
  --max-val-samples 8
```

Local Windows status check:

```powershell
D:\Apps\anaconda\envs\pytorch_env\python.exe scripts\check_foundation_model_status.py --project-root D:\vegetation_models_v2 --write-json results\foundation_model_status.json
```
