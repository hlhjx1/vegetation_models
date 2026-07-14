# Robustness Perturbation Results

- Dataset: 2024-seg fixed subset (n=32)
- Perturbations: clean, brightness, contrast, gaussian_blur, shadow_illumination
- Metrics: mIoU, F1, Accuracy, Delta_mIoU, Delta_F1, Delta_Accuracy

## Clean Baseline

| Model | mIoU | F1 | Accuracy |
|---|---:|---:|---:|
| SAM2.1-Tiny | 0.7624 | 0.9040 | 0.8757 |
| SegEarth-OV | 0.3895 | 0.6040 | 0.5638 |

## Robustness Summary

| Model | Mean Delta_mIoU | Worst Delta_mIoU | Worst condition |
|---|---:|---:|---|
| SAM2.1-Tiny | -0.0026 | -0.0207 | gaussian_blur:kernel_7 |
| SegEarth-OV | +0.0084 | -0.0291 | shadow_illumination:strong |

## Per-condition Results

| Model | Perturbation | Strength | mIoU | F1 | Accuracy | Delta_mIoU | Delta_F1 | Delta_Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SAM2.1-Tiny | clean | none | 0.7624 | 0.9040 | 0.8757 | +0.0000 | +0.0000 | +0.0000 |
| SAM2.1-Tiny | brightness | 0.70 | 0.7659 | 0.9053 | 0.8776 | +0.0035 | +0.0013 | +0.0019 |
| SAM2.1-Tiny | brightness | 0.85 | 0.7653 | 0.9051 | 0.8772 | +0.0028 | +0.0010 | +0.0015 |
| SAM2.1-Tiny | brightness | 1.15 | 0.7606 | 0.9036 | 0.8748 | -0.0018 | -0.0005 | -0.0009 |
| SAM2.1-Tiny | brightness | 1.30 | 0.7590 | 0.9032 | 0.8740 | -0.0034 | -0.0008 | -0.0016 |
| SAM2.1-Tiny | contrast | 0.75 | 0.7647 | 0.9048 | 0.8769 | +0.0023 | +0.0008 | +0.0012 |
| SAM2.1-Tiny | contrast | 0.90 | 0.7638 | 0.9045 | 0.8764 | +0.0013 | +0.0004 | +0.0007 |
| SAM2.1-Tiny | contrast | 1.10 | 0.7618 | 0.9039 | 0.8754 | -0.0006 | -0.0001 | -0.0003 |
| SAM2.1-Tiny | contrast | 1.25 | 0.7608 | 0.9037 | 0.8749 | -0.0016 | -0.0004 | -0.0008 |
| SAM2.1-Tiny | gaussian_blur | kernel_3 | 0.7531 | 0.9004 | 0.8704 | -0.0093 | -0.0037 | -0.0052 |
| SAM2.1-Tiny | gaussian_blur | kernel_5 | 0.7473 | 0.8976 | 0.8669 | -0.0151 | -0.0064 | -0.0088 |
| SAM2.1-Tiny | gaussian_blur | kernel_7 | 0.7417 | 0.8943 | 0.8631 | -0.0207 | -0.0098 | -0.0126 |
| SAM2.1-Tiny | shadow_illumination | mild | 0.7640 | 0.9046 | 0.8765 | +0.0016 | +0.0005 | +0.0008 |
| SAM2.1-Tiny | shadow_illumination | moderate | 0.7646 | 0.9048 | 0.8768 | +0.0022 | +0.0007 | +0.0011 |
| SAM2.1-Tiny | shadow_illumination | strong | 0.7648 | 0.9048 | 0.8769 | +0.0024 | +0.0007 | +0.0012 |
| SegEarth-OV | clean | none | 0.3895 | 0.6040 | 0.5638 | +0.0000 | +0.0000 | +0.0000 |
| SegEarth-OV | brightness | 0.70 | 0.4015 | 0.6255 | 0.5779 | +0.0121 | +0.0214 | +0.0141 |
| SegEarth-OV | brightness | 0.85 | 0.4040 | 0.6252 | 0.5798 | +0.0145 | +0.0212 | +0.0161 |
| SegEarth-OV | brightness | 1.15 | 0.3725 | 0.5924 | 0.5468 | -0.0170 | -0.0116 | -0.0170 |
| SegEarth-OV | brightness | 1.30 | 0.3832 | 0.6031 | 0.5580 | -0.0063 | -0.0009 | -0.0057 |
| SegEarth-OV | contrast | 0.75 | 0.4120 | 0.6324 | 0.5879 | +0.0226 | +0.0284 | +0.0242 |
| SegEarth-OV | contrast | 0.90 | 0.3977 | 0.6190 | 0.5734 | +0.0083 | +0.0150 | +0.0097 |
| SegEarth-OV | contrast | 1.10 | 0.3789 | 0.5991 | 0.5535 | -0.0106 | -0.0049 | -0.0102 |
| SegEarth-OV | contrast | 1.25 | 0.3638 | 0.5861 | 0.5378 | -0.0257 | -0.0179 | -0.0259 |
| SegEarth-OV | gaussian_blur | kernel_3 | 0.4083 | 0.5948 | 0.5803 | +0.0189 | -0.0092 | +0.0165 |
| SegEarth-OV | gaussian_blur | kernel_5 | 0.4604 | 0.6608 | 0.6324 | +0.0709 | +0.0568 | +0.0686 |
| SegEarth-OV | gaussian_blur | kernel_7 | 0.4745 | 0.6855 | 0.6475 | +0.0850 | +0.0815 | +0.0837 |
| SegEarth-OV | shadow_illumination | mild | 0.3880 | 0.6125 | 0.5639 | -0.0015 | +0.0085 | +0.0001 |
| SegEarth-OV | shadow_illumination | moderate | 0.3643 | 0.5846 | 0.5381 | -0.0252 | -0.0194 | -0.0257 |
| SegEarth-OV | shadow_illumination | strong | 0.3604 | 0.5835 | 0.5343 | -0.0291 | -0.0205 | -0.0294 |
