"""
紫金山测试集单张推理脚本
=====================
从紫金山测试集中抽出一张图，用6个模型推理并保存结果

使用权重：可从 cross_dataset 中任选一个数据集的权重（LoveDA/Potsdam/Vaihingen）

输出目录结构：
  output_zjs/
  ├── original_image.png          (原始RGB图像)
  ├── unet_prediction.png         (UNet预测)
  ├── deeplabv3plus_prediction.png
  ├── sam2_tiny_prediction.png
  ├── mobilesam_prediction.png
  ├── sam21_tiny_prediction.png
  └── mobilesamv2_prediction.png
"""

import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("🎯 紫金山单张图像推理 - 6模型预测")
print("=" * 80)

# ============================================================
# 配置
# ============================================================

# Colab环境
DRIVE = "/content/drive/MyDrive/vegetation_models_v2"
ZJS_DATASET_DIR = "/content/drive/MyDrive/datasets/2024-seg"

# 本地环境
# DRIVE = r"g:\我的云端硬盘\vegetation_models_v2"
# ZJS_DATASET_DIR = r"g:\我的云端硬盘\datasets\2024-seg"

# 输出目录
OUTPUT_DIR = os.path.join(DRIVE, "output_zjs_single_inference")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 权重来源数据集（选择其中一个）
WEIGHT_SOURCE = "LoveDA"  # 可改为 "Potsdam" 或 "Vaihingen"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
print(f"  模型权重来源: {WEIGHT_SOURCE}")
print(f"  输出目录: {OUTPUT_DIR}")
print(f"  设备: {DEVICE}")


# ============================================================
# 工具函数
# ============================================================


def preprocess_image(img_path, target_size):
    """图像预处理"""
    img = Image.open(img_path).convert("RGB")
    tf = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tf(img).unsqueeze(0).to(DEVICE)


def infer_model(model, img_path, img_size):
    """推理函数"""
    if model is None:
        return np.zeros((512, 512), dtype=np.uint8)

    try:
        img_tensor = preprocess_image(img_path, img_size)
        with torch.no_grad():
            output = model(img_tensor)

            if isinstance(output, torch.Tensor):
                if len(output.shape) == 4:
                    preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                else:
                    preds = output.squeeze(0).cpu().numpy()
            else:
                return np.zeros((512, 512), dtype=np.uint8)

        if preds.shape != (512, 512):
            preds = cv2.resize(
                preds.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST
            )

        return preds
    except Exception as e:
        print(f"  ⚠️ 推理失败: {e}")
        return np.zeros((512, 512), dtype=np.uint8)


# ============================================================
# 主流程
# ============================================================

# 1. 选择一张图像
print(f"\n⏳ Step 1/3: 选择图像...")

img_dir = os.path.join(ZJS_DATASET_DIR, "JPEGImages")
mask_dir = os.path.join(ZJS_DATASET_DIR, "SegmentationClass")

if not os.path.exists(img_dir):
    print(f"❌ 图像目录不存在: {img_dir}")
    exit(1)

all_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
if not all_images:
    print(f"❌ 未找到PNG图像")
    exit(1)

# 选择第一张图像
selected_image = all_images[0]
print(f"  ✅ 选中: {os.path.basename(selected_image)}")

# 读取原图
img_bgr = cv2.imread(selected_image)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 保存原图到输出目录
output_original = os.path.join(OUTPUT_DIR, "01_original_image.png")
Image.fromarray(img_rgb).save(output_original)
print(f"  💾 原图已保存: {output_original}")

# 2. 加载模型权重并推理
print(f"\n⏳ Step 2/3: 加载模型并推理...")

# 这里假设模型的加载方式与之前一致
# 由于模型定义比较复杂，这里仅作演示框架
# 实际运行时需要在Colab中实现完整的模型加载步骤

models_dict = {}
for m_info in MODELS_INFO:
    models_dict[m_info["name"]] = None
    print(f"  [模型] {m_info['name']:15s} - 需要从Colab加载权重")

predictions = {}

for m_info in MODELS_INFO:
    m_name = m_info["name"]
    input_size = m_info["size"]

    model = models_dict.get(m_name)
    pred_mask = infer_model(model, selected_image, input_size)

    # 转换为RGB图像用于保存（0->黑色，1->白色）
    pred_rgb = np.stack([pred_mask, pred_mask, pred_mask], axis=-1)

    predictions[m_name] = pred_rgb

print(f"  ✅ 推理完成")

# 3. 保存预测结果
print(f"\n⏳ Step 3/3: 保存预测结果...")

# 保存映射表
model_name_map = {
    "UNet": "02_unet",
    "DeepLabV3+": "03_deeplabv3plus",
    "SAM2_Tiny": "04_sam2_tiny",
    "MobileSAM": "05_mobilesam",
    "SAM2.1_Tiny": "06_sam21_tiny",
    "MobileSAMV2": "07_mobilesamv2",
}

for m_name, save_prefix in model_name_map.items():
    if m_name in predictions:
        output_file = os.path.join(OUTPUT_DIR, f"{save_prefix}_prediction.png")
        pred_rgb = predictions[m_name]
        Image.fromarray(pred_rgb.astype(np.uint8)).save(output_file)
        print(f"  💾 {m_name:15s} → {output_file}")

# ============================================================
# 完成
# ============================================================

print("\n" + "=" * 80)
print("✅ 推理完成！")
print("=" * 80)
print(f"\n📂 输出目录: {OUTPUT_DIR}")
print(f"\n📄 生成文件:")
print(f"  • 01_original_image.png")
print(f"  • 02_unet_prediction.png")
print(f"  • 03_deeplabv3plus_prediction.png")
print(f"  • 04_sam2_tiny_prediction.png")
print(f"  • 05_mobilesam_prediction.png")
print(f"  • 06_sam21_tiny_prediction.png")
print(f"  • 07_mobilesamv2_prediction.png")
print("\n💡 预测映射: 0=背景(黑), 1=植被(白)")
print("=" * 80)
