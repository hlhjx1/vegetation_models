# Robustness to Observation-Condition Perturbations

- Dataset: held-out `datasets/2024-seg` test subset (32 fixed images).
- Models: six models with valid fine-tuned checkpoints; no model was retrained.
- Metrics: global two-class mIoU, vegetation F1, and pixel accuracy calculated by this script on the same original-resolution ground-truth masks.
- Deltas: perturbed metric minus the clean metric for the same model; values are proportions, not percentage points.
- Shadow/illumination: deterministic synthetic cast shadows, seeded by filename and therefore identical for every model.

## Clean baseline and robustness summary

| Model | Clean mIoU | Clean F1 | Clean Accuracy | Mean Delta_mIoU | Worst Delta_mIoU | Worst perturbation |
|---|---:|---:|---:|---:|---:|---|
| SAM2-Tiny | 0.7657 | 0.9047 | 0.8772 | -0.0017 | -0.0122 | gaussian_blur, kernel_7 |
| MobileSAMV2 | 0.7492 | 0.8923 | 0.8648 | -0.0013 | -0.0303 | gaussian_blur, kernel_7 |
| YOLO11s-seg | 0.7465 | 0.8957 | 0.8655 | -0.0074 | -0.0573 | gaussian_blur, kernel_7 |
| UNet | 0.7390 | 0.8865 | 0.8582 | -0.0113 | -0.1023 | brightness, 1.30 |
| MobileSAM | 0.7186 | 0.8854 | 0.8499 | -0.0032 | -0.0352 | gaussian_blur, kernel_7 |
| DeepLabV3+ | 0.7167 | 0.8808 | 0.8467 | -0.0384 | -0.2416 | gaussian_blur, kernel_7 |

## Perturbation-category summary

| Perturbation category | Mean Delta_mIoU |
|---|---:|
| gaussian_blur | -0.0371 |
| brightness | -0.0070 |
| contrast | -0.0026 |
| shadow_illumination | +0.0005 |

## Paper-table suggestion

Use the clean-baseline robustness table as the compact manuscript table. If space allows, add the perturbation-category summary as a second short table.

Suggested paragraph:

> To examine adaptation to observation-condition changes, we conducted an inference-only robustness test using the same fixed 32-image held-out subset from the 2024 segmentation dataset. Brightness, contrast, Gaussian blur, and synthetic shadow/illumination perturbations were applied identically to all completed models, while ground-truth masks were unchanged. Among the completed models, MobileSAMV2 showed the smallest average mIoU degradation under perturbations (mean Delta_mIoU = -0.0013). The most challenging condition overall was gaussian_blur (mean Delta_mIoU = -0.0371).

## Revision response draft

> Response to Reviewer 2, Comment 4: Thank you for the helpful suggestion. We added a lightweight inference-only robustness analysis to evaluate adaptation to common observation-condition variations. Using the same fixed held-out subset, we applied brightness, contrast, Gaussian blur, and synthetic shadow/illumination perturbations to the input images and evaluated all completed models with the same pixel-level mIoU, F1, and accuracy implementation. The results show that MobileSAMV2 is comparatively stable across perturbations, whereas gaussian_blur is the most damaging observation change on average. We report these results in the robustness summary table and provide the experimental scripts and CSV outputs for reproducibility.

## Per-condition results

| Model | Perturbation | Strength | mIoU | F1 | Accuracy | Delta_mIoU | Delta_F1 | Delta_Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SAM2-Tiny | clean | none | 0.7657 | 0.9047 | 0.8772 | +0.0000 | +0.0000 | +0.0000 |
| SAM2-Tiny | brightness | 0.70 | 0.7642 | 0.9039 | 0.8762 | -0.0015 | -0.0008 | -0.0010 |
| SAM2-Tiny | brightness | 0.85 | 0.7654 | 0.9044 | 0.8769 | -0.0002 | -0.0003 | -0.0002 |
| SAM2-Tiny | brightness | 1.15 | 0.7658 | 0.9047 | 0.8772 | +0.0001 | -0.0001 | +0.0000 |
| SAM2-Tiny | brightness | 1.30 | 0.7654 | 0.9043 | 0.8768 | -0.0003 | -0.0004 | -0.0003 |
| SAM2-Tiny | contrast | 0.75 | 0.7675 | 0.9052 | 0.8781 | +0.0019 | +0.0005 | +0.0009 |
| SAM2-Tiny | contrast | 0.90 | 0.7665 | 0.9050 | 0.8776 | +0.0009 | +0.0003 | +0.0004 |
| SAM2-Tiny | contrast | 1.10 | 0.7656 | 0.9049 | 0.8772 | -0.0000 | +0.0002 | +0.0001 |
| SAM2-Tiny | contrast | 1.25 | 0.7657 | 0.9051 | 0.8774 | +0.0001 | +0.0004 | +0.0002 |
| SAM2-Tiny | gaussian_blur | kernel_3 | 0.7593 | 0.9029 | 0.8740 | -0.0064 | -0.0018 | -0.0032 |
| SAM2-Tiny | gaussian_blur | kernel_5 | 0.7564 | 0.9015 | 0.8722 | -0.0093 | -0.0033 | -0.0050 |
| SAM2-Tiny | gaussian_blur | kernel_7 | 0.7534 | 0.8990 | 0.8698 | -0.0122 | -0.0057 | -0.0073 |
| SAM2-Tiny | shadow_illumination | mild | 0.7667 | 0.9050 | 0.8777 | +0.0010 | +0.0003 | +0.0005 |
| SAM2-Tiny | shadow_illumination | moderate | 0.7667 | 0.9049 | 0.8776 | +0.0010 | +0.0002 | +0.0005 |
| SAM2-Tiny | shadow_illumination | strong | 0.7663 | 0.9047 | 0.8774 | +0.0007 | +0.0000 | +0.0002 |
| MobileSAM | clean | none | 0.7186 | 0.8854 | 0.8499 | +0.0000 | +0.0000 | +0.0000 |
| MobileSAM | brightness | 0.70 | 0.7190 | 0.8854 | 0.8501 | +0.0004 | +0.0001 | +0.0002 |
| MobileSAM | brightness | 0.85 | 0.7175 | 0.8851 | 0.8493 | -0.0011 | -0.0003 | -0.0006 |
| MobileSAM | brightness | 1.15 | 0.7223 | 0.8867 | 0.8520 | +0.0037 | +0.0013 | +0.0021 |
| MobileSAM | brightness | 1.30 | 0.7244 | 0.8868 | 0.8528 | +0.0058 | +0.0014 | +0.0029 |
| MobileSAM | contrast | 0.75 | 0.7139 | 0.8826 | 0.8466 | -0.0047 | -0.0028 | -0.0033 |
| MobileSAM | contrast | 0.90 | 0.7185 | 0.8853 | 0.8498 | -0.0002 | -0.0001 | -0.0001 |
| MobileSAM | contrast | 1.10 | 0.7171 | 0.8852 | 0.8492 | -0.0015 | -0.0002 | -0.0007 |
| MobileSAM | contrast | 1.25 | 0.7153 | 0.8850 | 0.8485 | -0.0033 | -0.0004 | -0.0014 |
| MobileSAM | gaussian_blur | kernel_3 | 0.7196 | 0.8860 | 0.8506 | +0.0010 | +0.0006 | +0.0007 |
| MobileSAM | gaussian_blur | kernel_5 | 0.7101 | 0.8809 | 0.8443 | -0.0085 | -0.0045 | -0.0056 |
| MobileSAM | gaussian_blur | kernel_7 | 0.6834 | 0.8691 | 0.8279 | -0.0352 | -0.0163 | -0.0221 |
| MobileSAM | shadow_illumination | mild | 0.7184 | 0.8852 | 0.8497 | -0.0003 | -0.0002 | -0.0002 |
| MobileSAM | shadow_illumination | moderate | 0.7184 | 0.8851 | 0.8497 | -0.0002 | -0.0003 | -0.0003 |
| MobileSAM | shadow_illumination | strong | 0.7172 | 0.8844 | 0.8489 | -0.0014 | -0.0010 | -0.0010 |
| MobileSAMV2 | clean | none | 0.7492 | 0.8923 | 0.8648 | +0.0000 | +0.0000 | +0.0000 |
| MobileSAMV2 | brightness | 0.70 | 0.7494 | 0.8924 | 0.8650 | +0.0002 | +0.0001 | +0.0001 |
| MobileSAMV2 | brightness | 0.85 | 0.7487 | 0.8921 | 0.8646 | -0.0005 | -0.0002 | -0.0003 |
| MobileSAMV2 | brightness | 1.15 | 0.7506 | 0.8930 | 0.8657 | +0.0014 | +0.0007 | +0.0008 |
| MobileSAMV2 | brightness | 1.30 | 0.7516 | 0.8935 | 0.8663 | +0.0024 | +0.0012 | +0.0015 |
| MobileSAMV2 | contrast | 0.75 | 0.7466 | 0.8907 | 0.8631 | -0.0026 | -0.0016 | -0.0018 |
| MobileSAMV2 | contrast | 0.90 | 0.7494 | 0.8924 | 0.8649 | +0.0002 | +0.0001 | +0.0001 |
| MobileSAMV2 | contrast | 1.10 | 0.7509 | 0.8936 | 0.8661 | +0.0017 | +0.0013 | +0.0013 |
| MobileSAMV2 | contrast | 1.25 | 0.7525 | 0.8950 | 0.8674 | +0.0033 | +0.0027 | +0.0026 |
| MobileSAMV2 | gaussian_blur | kernel_3 | 0.7589 | 0.8999 | 0.8722 | +0.0097 | +0.0076 | +0.0074 |
| MobileSAMV2 | gaussian_blur | kernel_5 | 0.7500 | 0.8965 | 0.8673 | +0.0008 | +0.0042 | +0.0024 |
| MobileSAMV2 | gaussian_blur | kernel_7 | 0.7189 | 0.8846 | 0.8496 | -0.0303 | -0.0077 | -0.0153 |
| MobileSAMV2 | shadow_illumination | mild | 0.7483 | 0.8919 | 0.8643 | -0.0009 | -0.0004 | -0.0006 |
| MobileSAMV2 | shadow_illumination | moderate | 0.7479 | 0.8917 | 0.8641 | -0.0013 | -0.0006 | -0.0008 |
| MobileSAMV2 | shadow_illumination | strong | 0.7471 | 0.8912 | 0.8635 | -0.0021 | -0.0011 | -0.0013 |
| YOLO11s-seg | clean | none | 0.7465 | 0.8957 | 0.8655 | +0.0000 | +0.0000 | +0.0000 |
| YOLO11s-seg | brightness | 0.70 | 0.7425 | 0.8938 | 0.8631 | -0.0040 | -0.0019 | -0.0024 |
| YOLO11s-seg | brightness | 0.85 | 0.7443 | 0.8946 | 0.8642 | -0.0021 | -0.0010 | -0.0013 |
| YOLO11s-seg | brightness | 1.15 | 0.7481 | 0.8963 | 0.8664 | +0.0016 | +0.0006 | +0.0009 |
| YOLO11s-seg | brightness | 1.30 | 0.7504 | 0.8971 | 0.8677 | +0.0039 | +0.0014 | +0.0022 |
| YOLO11s-seg | contrast | 0.75 | 0.7474 | 0.8957 | 0.8659 | +0.0009 | +0.0000 | +0.0004 |
| YOLO11s-seg | contrast | 0.90 | 0.7475 | 0.8960 | 0.8660 | +0.0011 | +0.0003 | +0.0005 |
| YOLO11s-seg | contrast | 1.10 | 0.7471 | 0.8961 | 0.8660 | +0.0006 | +0.0005 | +0.0005 |
| YOLO11s-seg | contrast | 1.25 | 0.7488 | 0.8972 | 0.8672 | +0.0023 | +0.0015 | +0.0017 |
| YOLO11s-seg | gaussian_blur | kernel_3 | 0.7346 | 0.8901 | 0.8583 | -0.0119 | -0.0056 | -0.0073 |
| YOLO11s-seg | gaussian_blur | kernel_5 | 0.7115 | 0.8802 | 0.8444 | -0.0349 | -0.0155 | -0.0211 |
| YOLO11s-seg | gaussian_blur | kernel_7 | 0.6891 | 0.8691 | 0.8300 | -0.0573 | -0.0266 | -0.0355 |
| YOLO11s-seg | shadow_illumination | mild | 0.7463 | 0.8955 | 0.8654 | -0.0002 | -0.0002 | -0.0001 |
| YOLO11s-seg | shadow_illumination | moderate | 0.7452 | 0.8949 | 0.8647 | -0.0012 | -0.0008 | -0.0009 |
| YOLO11s-seg | shadow_illumination | strong | 0.7447 | 0.8947 | 0.8644 | -0.0018 | -0.0010 | -0.0012 |
| UNet | clean | none | 0.7390 | 0.8865 | 0.8582 | +0.0000 | +0.0000 | +0.0000 |
| UNet | brightness | 0.70 | 0.7546 | 0.8990 | 0.8702 | +0.0156 | +0.0125 | +0.0121 |
| UNet | brightness | 0.85 | 0.7521 | 0.8953 | 0.8674 | +0.0131 | +0.0088 | +0.0092 |
| UNet | brightness | 1.15 | 0.6987 | 0.8591 | 0.8294 | -0.0403 | -0.0274 | -0.0288 |
| UNet | brightness | 1.30 | 0.6367 | 0.8106 | 0.7821 | -0.1023 | -0.0760 | -0.0760 |
| UNet | contrast | 0.75 | 0.7307 | 0.8788 | 0.8513 | -0.0083 | -0.0078 | -0.0068 |
| UNet | contrast | 0.90 | 0.7469 | 0.8903 | 0.8630 | +0.0079 | +0.0038 | +0.0049 |
| UNet | contrast | 1.10 | 0.7268 | 0.8795 | 0.8500 | -0.0122 | -0.0070 | -0.0081 |
| UNet | contrast | 1.25 | 0.7009 | 0.8628 | 0.8319 | -0.0381 | -0.0237 | -0.0263 |
| UNet | gaussian_blur | kernel_3 | 0.7550 | 0.8979 | 0.8698 | +0.0160 | +0.0114 | +0.0116 |
| UNet | gaussian_blur | kernel_5 | 0.7455 | 0.8923 | 0.8635 | +0.0065 | +0.0058 | +0.0053 |
| UNet | gaussian_blur | kernel_7 | 0.6911 | 0.8548 | 0.8242 | -0.0479 | -0.0317 | -0.0339 |
| UNet | shadow_illumination | mild | 0.7467 | 0.8916 | 0.8635 | +0.0077 | +0.0050 | +0.0054 |
| UNet | shadow_illumination | moderate | 0.7499 | 0.8938 | 0.8658 | +0.0109 | +0.0072 | +0.0077 |
| UNet | shadow_illumination | strong | 0.7515 | 0.8951 | 0.8671 | +0.0125 | +0.0086 | +0.0089 |
| DeepLabV3+ | clean | none | 0.7167 | 0.8808 | 0.8467 | +0.0000 | +0.0000 | +0.0000 |
| DeepLabV3+ | brightness | 0.70 | 0.6873 | 0.8760 | 0.8334 | -0.0294 | -0.0048 | -0.0133 |
| DeepLabV3+ | brightness | 0.85 | 0.7117 | 0.8810 | 0.8450 | -0.0049 | +0.0003 | -0.0017 |
| DeepLabV3+ | brightness | 1.15 | 0.7075 | 0.8740 | 0.8397 | -0.0091 | -0.0068 | -0.0070 |
| DeepLabV3+ | brightness | 1.30 | 0.6960 | 0.8657 | 0.8310 | -0.0207 | -0.0151 | -0.0157 |
| DeepLabV3+ | contrast | 0.75 | 0.7053 | 0.8761 | 0.8399 | -0.0114 | -0.0046 | -0.0067 |
| DeepLabV3+ | contrast | 0.90 | 0.7176 | 0.8813 | 0.8473 | +0.0009 | +0.0005 | +0.0006 |
| DeepLabV3+ | contrast | 1.10 | 0.7163 | 0.8809 | 0.8466 | -0.0004 | +0.0001 | -0.0001 |
| DeepLabV3+ | contrast | 1.25 | 0.7161 | 0.8812 | 0.8467 | -0.0006 | +0.0005 | +0.0000 |
| DeepLabV3+ | gaussian_blur | kernel_3 | 0.6394 | 0.8476 | 0.7988 | -0.0773 | -0.0332 | -0.0479 |
| DeepLabV3+ | gaussian_blur | kernel_5 | 0.5885 | 0.8238 | 0.7646 | -0.1282 | -0.0570 | -0.0821 |
| DeepLabV3+ | gaussian_blur | kernel_7 | 0.4750 | 0.7281 | 0.6605 | -0.2416 | -0.1526 | -0.1861 |
| DeepLabV3+ | shadow_illumination | mild | 0.7163 | 0.8816 | 0.8470 | -0.0003 | +0.0009 | +0.0003 |
| DeepLabV3+ | shadow_illumination | moderate | 0.7116 | 0.8805 | 0.8447 | -0.0050 | -0.0003 | -0.0020 |
| DeepLabV3+ | shadow_illumination | strong | 0.7064 | 0.8790 | 0.8419 | -0.0102 | -0.0018 | -0.0047 |

## Robustness ranking

Higher mean Delta_mIoU (i.e., a smaller average loss from the clean condition) indicates stronger robustness.

| Rank | Model | Mean Delta_mIoU over all 14 perturbation settings |
|---:|---|---:|
| 1 | MobileSAMV2 | -0.0013 |
| 2 | SAM2-Tiny | -0.0017 |
| 3 | MobileSAM | -0.0032 |
| 4 | YOLO11s-seg | -0.0074 |
| 5 | UNet | -0.0113 |
| 6 | DeepLabV3+ | -0.0384 |

## Perturbation sensitivity

More negative mean Delta_mIoU indicates a more damaging perturbation category, averaged over tested strengths and completed models.

| Perturbation | Mean Delta_mIoU |
|---|---:|
| gaussian_blur | -0.0371 |
| brightness | -0.0070 |
| contrast | -0.0026 |
| shadow_illumination | +0.0005 |

## Interpretation of Robustness Results

All six runnable fine-tuned models completed the final robustness experiment on the same fixed 32-image held-out subset. The clean-condition ranking by mIoU was SAM2-Tiny (0.7657), MobileSAMV2 (0.7492), YOLO11s-seg (0.7465), UNet (0.7390), MobileSAM (0.7186), and DeepLabV3+ (0.7167). This indicates that SAM2-Tiny achieved the strongest baseline segmentation accuracy on this subset, followed closely by MobileSAMV2 and YOLO11s-seg.

From the robustness perspective, MobileSAMV2 showed the smallest average degradation across all 14 perturbation settings (mean Delta_mIoU = -0.0013), followed by SAM2-Tiny (-0.0017) and MobileSAM (-0.0032). SAM2-Tiny combined the best clean mIoU with very stable behavior under brightness, contrast, and shadow/illumination changes; its largest drop occurred under kernel-7 Gaussian blur (Delta_mIoU = -0.0122), which is still small compared with the other models. MobileSAMV2 had a slightly lower clean mIoU than SAM2-Tiny, but its mean degradation was the smallest overall, suggesting strong adaptation to moderate observation-condition shifts.

YOLO11s-seg also showed competitive clean performance (mIoU = 0.7465), but it was more affected by Gaussian blur than the SAM-family lightweight models. Its kernel-7 blur setting produced a Delta_mIoU of -0.0573. UNet achieved a good clean mIoU after evaluating at the training-consistent 512x512 input size, but it was sensitive to strong over-brightening, with brightness 1.30 causing the largest UNet drop (Delta_mIoU = -0.1023). DeepLabV3+ was the least robust in this experiment, mainly because blur strongly degraded its predictions; kernel-7 blur reduced mIoU by -0.2416.

Across perturbation types, Gaussian blur was clearly the most challenging observation change, with an average Delta_mIoU of -0.0371 over all completed models. Brightness changes had a smaller average effect (-0.0070), and contrast changes were milder still (-0.0026). The synthetic shadow/illumination perturbation had almost no negative average impact (+0.0005), indicating that the completed models were generally stable under the tested deterministic shadow fields.

## Manuscript-Ready Summary

The robustness results show that the proposed SAM2-Tiny model achieved the highest clean-condition accuracy among the completed models (mIoU = 0.7657, F1 = 0.9047, Accuracy = 0.8772) and maintained stable performance under observation-condition perturbations (mean Delta_mIoU = -0.0017). MobileSAMV2 had the smallest average perturbation-induced degradation (mean Delta_mIoU = -0.0013), while SAM2-Tiny had the best combination of clean accuracy and robustness. Gaussian blur was the most damaging perturbation category, especially for DeepLabV3+ and YOLO11s-seg at stronger blur kernels. Brightness, contrast, and shadow/illumination changes had comparatively limited effects on the SAM-family models.

## Revision Response Draft

Thank you for the helpful suggestion. We added an inference-only robustness experiment to evaluate model adaptation to common observation-condition changes, including brightness variation, contrast variation, Gaussian blur, and synthetic shadow/illumination perturbation. All perturbations were applied to the same fixed held-out subset, and all completed models were evaluated using the same mIoU, F1, and pixel-accuracy implementation without retraining. The results show that SAM2-Tiny achieved the highest clean-condition mIoU (0.7657) and remained stable under perturbations (mean Delta_mIoU = -0.0017). MobileSAMV2 showed the smallest average degradation (mean Delta_mIoU = -0.0013). Gaussian blur was the most challenging perturbation overall, whereas brightness, contrast, and shadow/illumination changes caused smaller performance variations. These results have been added to the robustness summary and used to clarify the model behavior under different observation conditions.

## Supplement: Observation-Condition Robustness Plan for DINOv2, DINOv3, QwenVL, and LocateAnything

This section is appended for the newly added/current four-model experiments. It preserves the old robustness results above and does not change the old eight-model results. The planned experiment is inference-only: models are evaluated under perturbed input images without retraining, fine-tuning, or changing ground-truth masks.

### Current Result Status

| Model | Task type | Clean result source | Perturbation result status | Notes |
|---|---|---|---|---|
| DINOv2 ViT-L/14 + frozen segmentation head | frozen-head segmentation | `results/dinov2_four_dataset_results.csv`; heads in `9_DINOv2/four_dataset_runs/*/best_head.pth` | Pending for observation-condition perturbations | Can use pixel-level mIoU, vegetation IoU, F1, and accuracy under the same perturbations as the old robustness experiment |
| DINOv3 ViT-L/16 SAT-493M + frozen segmentation head | frozen-head segmentation | `results/dinov3_four_dataset_results.csv`; heads in `10_DINOv3/four_dataset_runs/*/best_head.pth` | Pending for observation-condition perturbations | Can be compared fairly with DINOv2 as a frozen-backbone segmentation model |
| Qwen/Qwen2.5-VL-3B-Instruct | prompt-based inference | `results/qwenvl_prompt_eval_results.csv` | Pending if running prompt-based perturbation inference | Should report box-mask IoU/F1/accuracy after converting generated boxes to masks; not a dense segmentation robustness result |
| nvidia/LocateAnything-3B | grounding | `results/locateanything_grounding_results.csv` | Pending if running grounding perturbation inference | Should report box-mask IoU/F1/accuracy and average box count; not a dense segmentation robustness result |

### Clean Four-Model Baseline Already Available

| Model | Dataset | n/eval samples | Metric type | IoU or mIoU | F1 | Accuracy | Extra |
|---|---|---:|---|---:|---:|---:|---|
| DINOv2 ViT-L/14 reg | Zijinshan | 67 | dense segmentation | 0.7466 mIoU / 0.8161 veg IoU | 0.8987 | 0.8673 | input 518 |
| DINOv2 ViT-L/14 reg | LoveDA | 100 | dense segmentation | 0.7076 mIoU / 0.4839 veg IoU | 0.6522 | 0.9355 | input 518 |
| DINOv2 ViT-L/14 reg | Potsdam | 80 | dense segmentation | 0.8403 mIoU / 0.8633 veg IoU | 0.9267 | 0.9152 | input 518 |
| DINOv2 ViT-L/14 reg | Vaihingen | 40 | dense segmentation | 0.8782 mIoU / 0.8616 veg IoU | 0.9256 | 0.9364 | input 518 |
| DINOv3 ViT-L/16 SAT-493M | Zijinshan | 67 | dense segmentation | 0.7683 mIoU / 0.8379 veg IoU | 0.9118 | 0.8822 | input 512 |
| DINOv3 ViT-L/16 SAT-493M | LoveDA | 100 | dense segmentation | 0.7505 mIoU / 0.5621 veg IoU | 0.7197 | 0.9433 | input 512 |
| DINOv3 ViT-L/16 SAT-493M | Potsdam | 80 | dense segmentation | 0.8530 mIoU / 0.8757 veg IoU | 0.9337 | 0.9227 | input 512 |
| DINOv3 ViT-L/16 SAT-493M | Vaihingen | 40 | dense segmentation | 0.8793 mIoU / 0.8619 veg IoU | 0.9258 | 0.9372 | input 512 |
| QwenVL2.5-3B | Zijinshan | 100 | prompt box-mask | 0.6212 IoU | 0.6981 | 0.7557 | avg boxes 1.15 |
| QwenVL2.5-3B | LoveDA | 100 | prompt box-mask | 0.1175 IoU | 0.1806 | 0.6437 | avg boxes 1.09 |
| QwenVL2.5-3B | Potsdam | 100 | prompt box-mask | 0.3482 IoU | 0.4233 | 0.6145 | avg boxes 0.73 |
| QwenVL2.5-3B | Vaihingen | 100 | prompt box-mask | 0.4615 IoU | 0.6080 | 0.5290 | avg boxes 1.11 |
| LocateAnything-3B | LoveDA | 100 | grounding box-mask | 0.1122 IoU | 0.1756 | 0.6649 | avg boxes 16.89 |
| LocateAnything-3B | Potsdam | 100 | grounding box-mask | 0.5038 IoU | 0.5914 | 0.6932 | avg boxes 2.74 |
| LocateAnything-3B | Vaihingen | 100 | grounding box-mask | 0.5444 IoU | 0.6924 | 0.7046 | avg boxes 8.85 |
| LocateAnything-3B | Zijinshan | 100 | grounding box-mask | 0.5747 IoU | 0.6457 | 0.7055 | avg boxes 13.64 |

### Recommended Perturbation Protocol

Use exactly the same train/eval split and ground-truth masks as the clean evaluation. Apply perturbations only to the input RGB image. Do not retrain the model and do not augment the training data for this analysis.

| Perturbation | Strengths | Implementation note |
|---|---|---|
| clean | none | Original image, unchanged |
| brightness | 0.7x, 1.3x | Multiply RGB intensity and clip to [0, 255] |
| contrast | 0.7x, 1.3x | Adjust contrast around the per-image mean or PIL midpoint |
| Gaussian blur | sigma/radius 1, 2 | Apply deterministic Gaussian blur to RGB only |
| Gaussian noise | sigma 0.03, 0.05 | Add seeded zero-mean Gaussian noise in [0, 1] scale, then clip |

For DINOv2 and DINOv3, report pixel-level dense segmentation metrics. For QwenVL2.5-3B and LocateAnything, report prompt/grounding box-mask metrics and average predicted boxes. Keep the two groups in separate tables.

### Dense Segmentation Perturbation Table Template

| Model | Dataset | Perturbation | Strength | n | mIoU | Veg IoU | F1 | Accuracy | Delta mIoU | Delta F1 | Delta Accuracy |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 ViT-L/14 reg | Zijinshan | clean | none | 67 | TBD | TBD | TBD | TBD | 0.0000 | 0.0000 | 0.0000 |
| DINOv2 ViT-L/14 reg | Zijinshan | brightness | 0.7x | 67 | pending | pending | pending | pending | pending | pending | pending |
| DINOv3 ViT-L/16 SAT-493M | Zijinshan | clean | none | 67 | TBD | TBD | TBD | TBD | 0.0000 | 0.0000 | 0.0000 |
| DINOv3 ViT-L/16 SAT-493M | Zijinshan | Gaussian blur | sigma 2 | 67 | pending | pending | pending | pending | pending | pending | pending |

### Prompt/Grounding Perturbation Table Template

| Model | Dataset | Perturbation | Strength | n | Box-mask IoU | Box-mask F1 | Box-mask Accuracy | Avg boxes | Delta IoU | Delta F1 | Delta Accuracy |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QwenVL2.5-3B | Zijinshan | clean | none | 100 | 0.6212 | 0.6981 | 0.7557 | 1.15 | 0.0000 | 0.0000 | 0.0000 |
| QwenVL2.5-3B | Zijinshan | brightness | 0.7x | 100 | pending | pending | pending | pending | pending | pending | pending |
| LocateAnything-3B | Zijinshan | clean | none | 100 | 0.5747 | 0.6457 | 0.7055 | 13.64 | 0.0000 | 0.0000 | 0.0000 |
| LocateAnything-3B | Zijinshan | Gaussian noise | sigma 0.05 | 100 | pending | pending | pending | pending | pending | pending | pending |

### Colab Script Skeleton

This skeleton first finds the code/weight locations in `/content`, then defines deterministic perturbations. DINOv2/DINOv3 require the local best heads to be uploaded or synced to the project-relative paths under `9_DINOv2/four_dataset_runs/*/best_head.pth` and `10_DINOv3/four_dataset_runs/*/best_head.pth`.

```python
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PROJECT = Path("/content/vegetation_models_v2")
BINARY_ROOT = Path("/content/binary")

required = {
    "dinov2_heads": [PROJECT / "9_DINOv2/four_dataset_runs" / d / "best_head.pth" for d in ["zijinshan", "loveda", "potsdam", "vaihingen"]],
    "dinov3_heads": [PROJECT / "10_DINOv3/four_dataset_runs" / d / "best_head.pth" for d in ["zijinshan", "loveda", "potsdam", "vaihingen"]],
    "dinov3_repo": [PROJECT / "10_DINOv3/code/dinov3-main"],
    "dinov3_weights": [PROJECT / "10_DINOv3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"],
    "qwen_cache_or_weights": [PROJECT / "11_QwenVL/weights", Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct")],
    "locateanything_cache_or_weights": [PROJECT / "12_LocatingAnything/weights", Path("/root/.cache/huggingface/hub/models--nvidia--LocateAnything-3B")],
}

print(json.dumps({
    k: {"found": [str(p) for p in v if p.exists()], "checked": [str(p) for p in v]}
    for k, v in required.items()
}, indent=2))

def perturb_image(img, perturbation, strength, seed=0):
    img = img.convert("RGB")
    if perturbation == "clean":
        return img
    if perturbation == "brightness":
        return ImageEnhance.Brightness(img).enhance(float(strength))
    if perturbation == "contrast":
        return ImageEnhance.Contrast(img).enhance(float(strength))
    if perturbation == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(strength)))
    if perturbation == "gaussian_noise":
        rng = np.random.default_rng(seed)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0.0, float(strength), arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255.0).astype(np.uint8))
    raise ValueError(f"Unknown perturbation: {perturbation}")

conditions = [
    ("clean", "none"),
    ("brightness", 0.7),
    ("brightness", 1.3),
    ("contrast", 0.7),
    ("contrast", 1.3),
    ("gaussian_blur", 1),
    ("gaussian_blur", 2),
    ("gaussian_noise", 0.03),
    ("gaussian_noise", 0.05),
]

# Dense DINO evaluation plan:
# 1. Reuse dataset loading and metric code from scripts/run_dinov2_four_dataset_train.py.
# 2. For each dataset and condition, perturb the input image after loading and before normalization.
# 3. Load the matching best_head.pth and run inference only.
# 4. Save rows with Model, Dataset, Perturbation, Strength, n, mIoU, Veg_IoU, F1, Accuracy, deltas.

# Prompt/grounding evaluation plan:
# 1. Reuse scripts/run_qwenvl_vegetation_prompt_eval.py and scripts/run_locateanything_vegetation_grounding.py.
# 2. Insert perturb_image before sending each PIL image to the processor/model.
# 3. Keep the same prompt and decoding settings for every condition.
# 4. Convert boxes to masks exactly as in the existing scripts and save box-mask metrics plus avg boxes.
```

### Reviewer 2 Comment 4 Response Text for the Four-Model Supplement

Thank you for the helpful suggestion. In addition to resolution-related evaluation, we prepared an inference-only observation-condition robustness protocol for the newly added DINOv2, DINOv3, QwenVL2.5-3B, and LocateAnything-3B experiments. The protocol applies controlled brightness, contrast, Gaussian blur, and Gaussian noise perturbations to the input images while keeping the ground-truth masks and model weights unchanged. DINOv2 and DINOv3 will be evaluated as frozen-backbone dense segmentation models using pixel-level mIoU, vegetation IoU, F1, and accuracy. QwenVL2.5-3B and LocateAnything-3B will be evaluated separately as prompt-based localization/grounding models using box-mask IoU, F1, accuracy, and average predicted box count. This design isolates robustness to observation-condition changes and does not involve retraining.
