# DINOv2 Frozen-Backbone Smoke Train

- Purpose: validate data loading, frozen DINOv2 feature extraction, lightweight segmentation-head training, mIoU calculation, and prediction-mask export.
- Dataset: `/content/vegetation_models_v2/datasets/2024-seg`.
- Train/val samples used: 12 / 8.
- Input size: 518. Epochs: 1. Batch size: 1.
- Final val loss: 0.5378. Final val mIoU: 0.5363. Final val F1: 0.8112.
- CSV log: `/content/vegetation_models_v2/results/dinov2_smoke_train_log.csv`.

This smoke run is not a formal long-training result.
