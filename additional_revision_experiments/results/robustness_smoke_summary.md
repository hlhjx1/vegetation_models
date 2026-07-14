# Robustness to Observation-Condition Perturbations

- Dataset: held-out `datasets/2024-seg` test subset (4 fixed images).
- Models: six models with valid fine-tuned checkpoints; no model was retrained.
- Metrics: global two-class mIoU, vegetation F1, and pixel accuracy calculated by this script on the same original-resolution ground-truth masks.
- Deltas: perturbed metric minus the clean metric for the same model; values are proportions, not percentage points.
- Shadow/illumination: deterministic synthetic cast shadows, seeded by filename and therefore identical for every model.

## Clean baseline and robustness summary

| Model | Clean mIoU | Clean F1 | Clean Accuracy | Mean Delta_mIoU | Worst Delta_mIoU | Worst perturbation |
|---|---:|---:|---:|---:|---:|---|
| YOLO11s-seg | 0.7926 | 0.8188 | 0.9132 | -0.0227 | -0.0829 | gaussian_blur, kernel_7 |
| SAM2-Tiny | 0.7806 | 0.8091 | 0.9055 | -0.0034 | -0.0069 | brightness, 0.70 |
| UNet | 0.7791 | 0.8041 | 0.9075 | -0.0764 | -0.1851 | gaussian_blur, kernel_7 |
| DeepLabV3+ | 0.7774 | 0.8001 | 0.9085 | -0.1359 | -0.3819 | gaussian_blur, kernel_7 |
| MobileSAMV2 | 0.7463 | 0.7632 | 0.8963 | -0.0192 | -0.0322 | contrast, 0.75 |
| MobileSAM | 0.7072 | 0.7299 | 0.8683 | -0.0307 | -0.0644 | gaussian_blur, kernel_7 |

## Perturbation-category summary

| Perturbation category | Mean Delta_mIoU |
|---|---:|
| gaussian_blur | -0.1223 |
| contrast | -0.0385 |
| brightness | -0.0167 |
| shadow_illumination | -0.0147 |

## Paper-table suggestion

Use the clean-baseline robustness table as the compact manuscript table. If space allows, add the perturbation-category summary as a second short table.

Suggested paragraph:

> To examine adaptation to observation-condition changes, we conducted an inference-only robustness test using the same fixed 4-image held-out subset from the 2024 segmentation dataset. Brightness, contrast, Gaussian blur, and synthetic shadow/illumination perturbations were applied identically to all completed models, while ground-truth masks were unchanged. Among the completed models, SAM2-Tiny showed the smallest average mIoU degradation under perturbations (mean Delta_mIoU = -0.0034). The most challenging condition overall was gaussian_blur (mean Delta_mIoU = -0.1223).

## Revision response draft

> Response to Reviewer 2, Comment 4: Thank you for the helpful suggestion. We added a lightweight inference-only robustness analysis to evaluate adaptation to common observation-condition variations. Using the same fixed held-out subset, we applied brightness, contrast, Gaussian blur, and synthetic shadow/illumination perturbations to the input images and evaluated all completed models with the same pixel-level mIoU, F1, and accuracy implementation. The results show that SAM2-Tiny is comparatively stable across perturbations, whereas gaussian_blur is the most damaging observation change on average. We report these results in the robustness summary table and provide the experimental scripts and CSV outputs for reproducibility.

## Per-condition results

| Model | Perturbation | Strength | mIoU | F1 | Accuracy | Delta_mIoU | Delta_F1 | Delta_Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SAM2-Tiny | clean | none | 0.7806 | 0.8091 | 0.9055 | +0.0000 | +0.0000 | +0.0000 |
| SAM2-Tiny | brightness | 0.70 | 0.7736 | 0.8031 | 0.9012 | -0.0069 | -0.0060 | -0.0043 |
| SAM2-Tiny | contrast | 0.75 | 0.7822 | 0.8100 | 0.9068 | +0.0016 | +0.0010 | +0.0013 |
| SAM2-Tiny | gaussian_blur | kernel_7 | 0.7743 | 0.8032 | 0.9019 | -0.0063 | -0.0058 | -0.0036 |
| SAM2-Tiny | shadow_illumination | strong | 0.7786 | 0.8071 | 0.9045 | -0.0019 | -0.0020 | -0.0009 |
| MobileSAM | clean | none | 0.7072 | 0.7299 | 0.8683 | +0.0000 | +0.0000 | +0.0000 |
| MobileSAM | brightness | 0.70 | 0.6925 | 0.7171 | 0.8574 | -0.0148 | -0.0128 | -0.0109 |
| MobileSAM | contrast | 0.75 | 0.6776 | 0.6994 | 0.8495 | -0.0296 | -0.0305 | -0.0188 |
| MobileSAM | gaussian_blur | kernel_7 | 0.6429 | 0.6597 | 0.8280 | -0.0644 | -0.0702 | -0.0403 |
| MobileSAM | shadow_illumination | strong | 0.6934 | 0.7129 | 0.8616 | -0.0138 | -0.0170 | -0.0067 |
| MobileSAMV2 | clean | none | 0.7463 | 0.7632 | 0.8963 | +0.0000 | +0.0000 | +0.0000 |
| MobileSAMV2 | brightness | 0.70 | 0.7356 | 0.7487 | 0.8931 | -0.0107 | -0.0146 | -0.0031 |
| MobileSAMV2 | contrast | 0.75 | 0.7141 | 0.7211 | 0.8845 | -0.0322 | -0.0421 | -0.0118 |
| MobileSAMV2 | gaussian_blur | kernel_7 | 0.7333 | 0.7587 | 0.8822 | -0.0130 | -0.0046 | -0.0141 |
| MobileSAMV2 | shadow_illumination | strong | 0.7252 | 0.7372 | 0.8877 | -0.0211 | -0.0260 | -0.0086 |
| YOLO11s-seg | clean | none | 0.7926 | 0.8188 | 0.9132 | +0.0000 | +0.0000 | +0.0000 |
| YOLO11s-seg | brightness | 0.70 | 0.7852 | 0.8115 | 0.9095 | -0.0075 | -0.0073 | -0.0037 |
| YOLO11s-seg | contrast | 0.75 | 0.7977 | 0.8237 | 0.9158 | +0.0051 | +0.0049 | +0.0026 |
| YOLO11s-seg | gaussian_blur | kernel_7 | 0.7097 | 0.7332 | 0.8693 | -0.0829 | -0.0857 | -0.0439 |
| YOLO11s-seg | shadow_illumination | strong | 0.7871 | 0.8133 | 0.9105 | -0.0055 | -0.0055 | -0.0027 |
| UNet | clean | none | 0.7791 | 0.8041 | 0.9075 | +0.0000 | +0.0000 | +0.0000 |
| UNet | brightness | 0.70 | 0.7694 | 0.7965 | 0.9008 | -0.0097 | -0.0075 | -0.0067 |
| UNet | contrast | 0.75 | 0.6681 | 0.6578 | 0.8659 | -0.1110 | -0.1462 | -0.0417 |
| UNet | gaussian_blur | kernel_7 | 0.5940 | 0.5502 | 0.8285 | -0.1851 | -0.2538 | -0.0790 |
| UNet | shadow_illumination | strong | 0.7793 | 0.8050 | 0.9071 | +0.0002 | +0.0010 | -0.0005 |
| DeepLabV3+ | clean | none | 0.7774 | 0.8001 | 0.9085 | +0.0000 | +0.0000 | +0.0000 |
| DeepLabV3+ | brightness | 0.70 | 0.7269 | 0.7582 | 0.8742 | -0.0506 | -0.0419 | -0.0344 |
| DeepLabV3+ | contrast | 0.75 | 0.7129 | 0.7240 | 0.8805 | -0.0646 | -0.0761 | -0.0280 |
| DeepLabV3+ | gaussian_blur | kernel_7 | 0.3956 | 0.4414 | 0.5880 | -0.3819 | -0.3587 | -0.3205 |
| DeepLabV3+ | shadow_illumination | strong | 0.7311 | 0.7516 | 0.8845 | -0.0464 | -0.0485 | -0.0240 |

## Robustness ranking

Higher mean Delta_mIoU (i.e., a smaller average loss from the clean condition) indicates stronger robustness.

| Rank | Model | Mean Delta_mIoU over all 14 perturbation settings |
|---:|---|---:|
| 1 | SAM2-Tiny | -0.0034 |
| 2 | MobileSAMV2 | -0.0192 |
| 3 | YOLO11s-seg | -0.0227 |
| 4 | MobileSAM | -0.0307 |
| 5 | UNet | -0.0764 |
| 6 | DeepLabV3+ | -0.1359 |

## Perturbation sensitivity

More negative mean Delta_mIoU indicates a more damaging perturbation category, averaged over tested strengths and completed models.

| Perturbation | Mean Delta_mIoU |
|---|---:|
| gaussian_blur | -0.1223 |
| contrast | -0.0385 |
| brightness | -0.0167 |
| shadow_illumination | -0.0147 |
