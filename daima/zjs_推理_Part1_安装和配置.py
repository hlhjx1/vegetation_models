# ==============================================================================
# Part 1: 安装依赖、配置路径、修补torch（第1个 Colab Cell）
# ==============================================================================

print("⏳ Part 1/3: 安装和配置...")

import os
import sys
import glob
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import warnings
import pickle

warnings.filterwarnings("ignore")

# ==================== 关键：torch.load全局修补 - 允许numpy数据结构 ====================
import torch.serialization

_original_load = torch.load


def _patched_load(f, *args, **kwargs):
    """允许加载包含numpy/复杂类型的权重 - 解决recursive depth错误"""
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(f, *args, **kwargs)


torch.load = _patched_load
print("✅ torch.load 全局修补完成 (weights_only=False)")
# ========================================================================

# 配置全局变量
DRIVE = "/content/drive/MyDrive/vegetation_models_v2"
ZJS_DATASET_DIR = "/content/drive/MyDrive/datasets/2024-seg"
OUTPUT_DIR = os.path.join(DRIVE, "output_zjs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 当前设备: {DEVICE}")
if DEVICE == "cuda":
    print(f"✅ GPU型号: {torch.cuda.get_device_name()}")

# ============ 配置sys.path（必须在这里配置，Part 2会使用）============
print(f"\n⏳ 配置模型代码路径...")

_sam2_code_path = os.path.join(DRIVE, "1_SAM2_Tiny/code")
_mobilesam_code_path = os.path.join(DRIVE, "2_MobileSAM/code")
_mobilev2_code_path = os.path.join(DRIVE, "6_MobileSAMV2/code")
_sam21_code_path = os.path.join(DRIVE, "5_SAM21_Tiny/code")

for path in [
    _sam2_code_path,
    _mobilesam_code_path,
    _mobilev2_code_path,
    _sam21_code_path,
]:
    if path not in sys.path:
        sys.path.insert(0, path)

print(f"✅ sys.path 配置完成")

# 6个模型配置
MODELS_INFO = [
    {"name": "UNet", "size": 512},
    {"name": "DeepLabV3+", "size": 512},
    {"name": "SAM2_Tiny", "size": 1024},
    {"name": "MobileSAM", "size": 1024},
    {"name": "SAM2.1_Tiny", "size": 1024},
    {"name": "MobileSAMV2", "size": 1024},
]

print(f"\n📦 配置信息:")
print(f"  紫金山数据集: {ZJS_DATASET_DIR}")
print(f"  输出目录: {OUTPUT_DIR}")
print(f"  设备: {DEVICE}")
print(f"  PyTorch版本: {torch.__version__}")
print(f"\n✅ Part 1 完成！下一步运行 zjs_推理_Part2_定义函数.py")
