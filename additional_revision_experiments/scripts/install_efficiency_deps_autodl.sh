#!/usr/bin/env bash
set -euo pipefail

# Run from the AutoDL project root. This installs only benchmark dependencies;
# it neither downloads public datasets nor trains/changes model checkpoints.
python -m pip install --upgrade pip
# Keep pkg_resources available for Ultralytics/OpenMMLab-era dependencies.
# The benchmark script adds the Python 3.12 pkgutil.ImpImporter shim before
# these packages are imported.
python -m pip install 'setuptools==60.2.0'
python -m pip install thop hydra-core omegaconf timm segmentation-models-pytorch ultralytics iopath

# SegEarth-OV's official evaluation stack. mmcv-lite supplies the Python API
# needed by this inference path and avoids compiling unrelated OpenMMLab CUDA ops.
python -m pip install mmengine mmsegmentation ftfy regex einops fairscale safetensors openpyxl
python -m pip install 'mmcv-lite==2.1.0'

# Required by SegEarth's JBU/SimFeatUp adaptive-convolution forward path.
# This invokes the upstream package build for the active PyTorch/CUDA image.
python -m pip install 'git+https://github.com/mhamilton723/FeatUp.git'

echo 'Dependencies installed. If FeatUp compilation fails, retain its full terminal log: SegEarth-OV will remain N/A rather than being replaced by a non-equivalent CLIP-only timing.'
