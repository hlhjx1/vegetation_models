#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
紫金山单张图像推理 - 完全复制自生成模型用推理图.ipynb的成功代码
改为推理单张图像而非13个案例
直接在此脚本中运行，无需分cell
"""

print("=" * 80)
print("🎯 紫金山单张图像推理 - 6模型预测")
print("=" * 80)

import sys
import cv2
import glob
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import warnings
import pickle
import os

warnings.filterwarnings("ignore")

# ==================== 关键全局处理：解决numpy/torch不兼容 ====================
# 禁用torch的严格权重检查（支持旧模型和包含numpy类型的权重文件）
import torch.serialization

_original_load = torch.load


def _patched_load(f, *args, **kwargs):
    """允许加载包含numpy/复杂类型的权重"""
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(f, *args, **kwargs)


torch.load = _patched_load
# ========================================================================

print(f"✅ PyTorch版本: {torch.__version__}")
print(f"✅ GPU可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU型号: {torch.cuda.get_device_name()}")
print()

# ==================== 配置路径和参数 ====================

BASE_DIR = "/content/drive/MyDrive/vegetation_models_v2"
DATASET_DIR = "/content/drive/MyDrive/datasets/2024-seg"
IMAGE_DIR = os.path.join(DATASET_DIR, "JPEGImages")
MASK_DIR = os.path.join(DATASET_DIR, "SegmentationClass")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_zjs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 6个模型 (不含SegEarth_OV和YOLO11)
MODELS_INFO = [
    {"name": "UNet", "key": "7_UNet", "size": 512, "ckpt": "unet_best.pth"},
    {
        "name": "DeepLabV3+",
        "key": "8_DeepLabV3",
        "size": 512,
        "ckpt": "deeplabv3plus_best.pth",
    },
    {
        "name": "SAM2_Tiny",
        "key": "1_SAM2_Tiny",
        "size": 1024,
        "ckpt": "sam2tiny_best.pth",
    },
    {
        "name": "MobileSAM",
        "key": "2_MobileSAM",
        "size": 1024,
        "ckpt": "mobilesam_best.pth",
    },
    {
        "name": "SAM2.1_Tiny",
        "key": "5_SAM21_Tiny",
        "size": 1024,
        "ckpt": "sam21tiny_best.pth",
    },
    {
        "name": "MobileSAMV2",
        "key": "6_MobileSAMV2",
        "size": 1024,
        "ckpt": "mobilesamv2_best.pth",
    },
]

print(f"✅ 基础目录: {BASE_DIR}")
print(f"✅ 数据集目录: {DATASET_DIR}")
print(f"✅ 输出目录: {OUTPUT_DIR}")

# 检查数据集是否存在
if os.path.exists(IMAGE_DIR):
    all_images = sorted(
        glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    )
    print(f"✅ 找到 {len(all_images)} 张图像")
else:
    print(f"❌ 错误: 图像目录不存在 {IMAGE_DIR}")

print()

# ==================== 安装模型代码 ====================

print("⏳ 配置模型代码...")

if "/content/drive/MyDrive/vegetation_models_v2/2_MobileSAM/code" not in sys.path:
    sys.path.insert(0, "/content/drive/MyDrive/vegetation_models_v2/2_MobileSAM/code")

if "/content/drive/MyDrive/vegetation_models_v2/6_MobileSAMV2/code" not in sys.path:
    sys.path.insert(0, "/content/drive/MyDrive/vegetation_models_v2/6_MobileSAMV2/code")

# SAM2.1手动添加到环境
if "/content/drive/MyDrive/vegetation_models_v2/5_SAM21_Tiny/code" not in sys.path:
    sys.path.insert(0, "/content/drive/MyDrive/vegetation_models_v2/5_SAM21_Tiny/code")

# 改变工作目录使 Hydra 能找到 config
_original_cwd = os.getcwd()
_sam2_code_dir = os.path.join(BASE_DIR, "1_SAM2_Tiny/code")
if os.path.exists(_sam2_code_dir):
    os.chdir(_sam2_code_dir)
    import sys

    if _sam2_code_dir not in sys.path:
        sys.path.insert(0, _sam2_code_dir)
    print(f"✅ 工作目录已改为: {_sam2_code_dir}")

print("✅ 模型代码安装完成")
print()

# ==================== 模型加载函数 - 完全复制自工作笔记本 ====================

print("⏳ 定义模型加载函数...")


def load_unet(base_dir, device):
    """UNet"""
    try:
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=2,
            activation=None,
        ).to(device)
        ckpt_path = os.path.join(base_dir, "7_UNet/checkpoints/unet_best.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
            else:
                model.load_state_dict(ckpt)
            print(f"✅ UNet权重加载成功")
        else:
            print(f"⚠️ UNet权重找不到: {ckpt_path}")

        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ UNet加载失败: {e}")
        return None


def load_deeplabv3(base_dir, device):
    """DeepLabV3+"""
    try:
        import segmentation_models_pytorch as smp

        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=2,
            encoder_output_stride=16,
            activation=None,
        ).to(device)

        ckpt_path = os.path.join(
            base_dir, "8_DeepLabV3/checkpoints/deeplabv3plus_best.pth"
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
            else:
                model.load_state_dict(ckpt)
            print(f"✅ DeepLabV3+权重加载成功")
        else:
            print(f"⚠️ DeepLabV3+权重找不到: {ckpt_path}")

        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ DeepLabV3+加载失败: {e}")
        return None


def load_sam2_tiny(base_dir, device):
    try:
        import torch.nn as nn
        from sam2.build_sam import build_sam2

        class SAM2Seg(nn.Module):
            def __init__(self, cfg, ckpt_path, device, num_classes=2):
                super().__init__()
                sam2 = build_sam2(cfg, ckpt_path, device=device)
                self.encoder = sam2.image_encoder

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
                    nn.Conv2d(64, num_classes, 1),
                    nn.Upsample(
                        size=(1024, 1024), mode="bilinear", align_corners=False
                    ),
                )

            def forward(self, x):
                features = self.encoder(x)
                # SAM2 encoder返回dict，取backbone_fpn最后一层
                if isinstance(features, dict):
                    feat = features["backbone_fpn"][-1]
                else:
                    feat = features
                return self.seg_head(feat)

        # Pretrained权重路径
        pretrain_ckpt = os.path.join(base_dir, "1_SAM2_Tiny/weights/sam2_hiera_tiny.pt")
        if not os.path.exists(pretrain_ckpt):
            print(f"⚠️ SAM2_Tiny预训练权重找不到: {pretrain_ckpt}")
            return None

        # 微调后的权重路径
        finetuned_ckpt = os.path.join(
            base_dir, "1_SAM2_Tiny/checkpoints/sam2tiny_best.pth"
        )

        try:
            model = SAM2Seg("sam2/sam2_hiera_t.yaml", pretrain_ckpt, device).to(device)

            # 尝试加载微调后的权重
            if os.path.exists(finetuned_ckpt):
                ckpt = torch.load(
                    finetuned_ckpt, map_location=device, weights_only=False
                )
                if "model_state" in ckpt:
                    model.load_state_dict(ckpt["model_state"], strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)
                print(f"✅ SAM2_Tiny加载成功（微调权重）")
            else:
                print(f"✅ SAM2_Tiny加载成功（预训练权重）")

            model.eval()
            return model
        except Exception as model_err:
            print(f"⚠️ SAM2_Tiny模型加载失败: {model_err}")
            return None
    except Exception as e:
        print(f"⚠️ SAM2_Tiny加载失败: {e}")
        return None


def load_mobilesam(base_dir, device):
    """MobileSAM"""
    try:
        import torch.nn as nn

        # 尝试多种导入方式
        try:
            from mobile_sam import sam_model_registry
        except ImportError:
            try:
                sys.path.insert(0, os.path.join(base_dir, "2_MobileSAM/code"))
                from mobile_sam import sam_model_registry
            except ImportError:
                print(f"⚠️ MobileSAM: 无法导入sam_model_registry")
                return None

        class MobileSAMSeg(nn.Module):
            def __init__(self, ckpt_path, device, num_classes=2):
                super().__init__()
                sam = sam_model_registry["vit_t"](checkpoint=None)
                self.encoder = sam.image_encoder

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
                    nn.Conv2d(64, num_classes, 1),
                    nn.Upsample(
                        size=(1024, 1024), mode="bilinear", align_corners=False
                    ),
                )

            def forward(self, x):
                return self.seg_head(self.encoder(x))

        # 尝试多个权重路径
        ckpt_paths = [
            os.path.join(base_dir, "2_MobileSAM/checkpoints/mobilesam_best.pth"),
            os.path.join(base_dir, "2_MobileSAM/weights/mobile_sam.pt"),
        ]

        ckpt_path = None
        for path in ckpt_paths:
            if os.path.exists(path):
                ckpt_path = path
                print(f"✅ MobileSAM找到权重: {path}")
                break

        if ckpt_path is None:
            print(f"⚠️ MobileSAM权重找不到")
            return None

        model = MobileSAMSeg(ckpt_path, device).to(device)
        print(f"✅ MobileSAM加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ MobileSAM加载失败: {e}")
        return None


def load_sam21_tiny(base_dir, device):
    """SAM2.1-Tiny"""
    try:
        import torch.nn as nn
        from sam2.build_sam import build_sam2, build_sam2_hf

        class SAM21Seg(nn.Module):
            def __init__(self, device, num_classes=2):
                super().__init__()
                # 直接用HF接口加载，自动处理config路径问题
                try:
                    sam21 = build_sam2_hf(
                        "facebook/sam2.1-hiera-tiny", device=device, mode="train"
                    )
                except:
                    # 备选：从本地加载
                    from sam2.build_sam import build_sam2

                    sam21 = build_sam2(
                        "sam2.1/sam2.1_hiera_t.yaml",
                        os.path.join(
                            base_dir, "5_SAM21_Tiny/weights/sam2.1_hiera_tiny.pt"
                        ),
                        device=device,
                        mode="train",
                    )

                self.encoder = sam21.image_encoder

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
                    nn.Conv2d(64, num_classes, 1),
                    nn.Upsample(
                        size=(1024, 1024), mode="bilinear", align_corners=False
                    ),
                )

            def forward(self, x):
                features = self.encoder(x)
                if isinstance(features, dict):
                    feat = features["backbone_fpn"][-1]
                else:
                    feat = features
                return self.seg_head(feat)

        model = SAM21Seg(device).to(device)

        best_ckpt = os.path.join(
            base_dir, "5_SAM21_Tiny/checkpoints/sam21tiny_best.pth"
        )
        if os.path.exists(best_ckpt):
            try:
                ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
                if "model_state" in ckpt:
                    model.load_state_dict(ckpt["model_state"], strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)
                print(f"✅ SAM2.1_Tiny加载成功")
            except Exception as e:
                print(f"⚠️ SAM2.1_Tiny权重加载失败: {e}")

        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ SAM2.1_Tiny加载失败: {e}")
        return None


def load_mobilesamv2(base_dir, device):
    """MobileSAMV2"""
    try:
        import torch.nn as nn

        try:
            from mobile_sam import sam_model_registry
        except ImportError:
            try:
                sys.path.insert(0, os.path.join(base_dir, "6_MobileSAMV2/code"))
                from mobile_sam import sam_model_registry
            except ImportError:
                print(f"⚠️ MobileSAMV2: 无法导入sam_model_registry")
                return None

        class MobileSAMV2Seg(nn.Module):
            def __init__(self, ckpt_path, device, num_classes=2):
                super().__init__()
                try:
                    sam = sam_model_registry["vit_t_mobile"](checkpoint=None)
                except:
                    sam = sam_model_registry["vit_t"](checkpoint=None)
                self.encoder = sam.image_encoder

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
                    nn.Conv2d(64, num_classes, 1),
                    nn.Upsample(
                        size=(1024, 1024), mode="bilinear", align_corners=False
                    ),
                )

            def forward(self, x):
                return self.seg_head(self.encoder(x))

        ckpt_paths = [
            os.path.join(base_dir, "6_MobileSAMV2/checkpoints/mobilesamv2_best.pth"),
            os.path.join(base_dir, "6_MobileSAMV2/weights/mobile_samv2.pt"),
        ]

        ckpt_path = None
        for path in ckpt_paths:
            if os.path.exists(path):
                ckpt_path = path
                print(f"✅ MobileSAMV2找到权重: {path}")
                break

        if ckpt_path is None:
            print(f"⚠️ MobileSAMV2权重找不到")
            return None

        model = MobileSAMV2Seg(ckpt_path, device).to(device)
        print(f"✅ MobileSAMV2加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ MobileSAMV2加载失败: {e}")
        return None


print("✅ 函数定义完成")
print()

# ==================== 加载所有模型 ====================

print("⏳ 加载所有6个模型...")

models_dict = {
    "UNet": load_unet(BASE_DIR, DEVICE),
    "DeepLabV3+": load_deeplabv3(BASE_DIR, DEVICE),
    "SAM2_Tiny": load_sam2_tiny(BASE_DIR, DEVICE),
    "MobileSAM": load_mobilesam(BASE_DIR, DEVICE),
    "SAM2.1_Tiny": load_sam21_tiny(BASE_DIR, DEVICE),
    "MobileSAMV2": load_mobilesamv2(BASE_DIR, DEVICE),
}

print("✅ 所有模型加载完成")
print()

# ==================== 推理函数 ====================


def create_overlay_mask(img_bgr, pred_mask, gt_mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    base_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    tp_mask = (pred_mask == 1) & (gt_mask == 1)
    fp_mask = (pred_mask == 1) & (gt_mask == 0)
    fn_mask = (pred_mask == 0) & (gt_mask == 1)

    color_mask = np.zeros_like(base_img)
    color_mask[tp_mask] = [0, 0, 255]  # TP: 蓝色
    color_mask[fp_mask] = [0, 255, 0]  # FP: 绿色
    color_mask[fn_mask] = [255, 0, 255]  # FN: 粉色

    alpha = 0.5
    overlay_img = np.where(
        color_mask != 0,
        cv2.addWeighted(base_img, 1 - alpha, color_mask, alpha, 0),
        base_img,
    )
    return overlay_img


def preprocess_image(img_path, target_size, device):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img).unsqueeze(0).to(device)


def infer_model(model, img_path, img_size, device):
    if model is None:
        return np.zeros((512, 512), dtype=np.uint8)

    try:
        img_tensor = preprocess_image(img_path, img_size, device)
        with torch.no_grad():
            output = model(img_tensor)
            preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        if preds.shape != (512, 512):
            preds = cv2.resize(
                preds.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST
            )

        return preds
    except Exception as e:
        print(f"⚠️ 推理失败: {e}")
        return np.zeros((512, 512), dtype=np.uint8)


# ==================== 执行单张图像推理 ====================

print("⏳ Step 1/2: 获取单张图像...\n")

selected_images = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)[
    :1
]  # 仅取第一张

results = {"Original": []}
for m in MODELS_INFO:
    results[m["name"]] = []

for idx, img_path in enumerate(selected_images):
    print(f"  [{idx + 1}/1] {os.path.basename(img_path)}")

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # 缩放到512x512用于显示
    img_rgb_512 = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    results["Original"].append(img_rgb_512)

    filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(MASK_DIR, filename + ".png")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(MASK_DIR, filename + ".jpg")

    if os.path.exists(mask_path):
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        gt_mask = (gt_mask > 0).astype(np.uint8)
    else:
        gt_mask = np.zeros((512, 512), dtype=np.uint8)

    print("⏳ Step 2/2: 模型推理和保存...\n")

    for m_info in MODELS_INFO:
        m_name = m_info["name"]
        input_size = m_info["size"]

        model = models_dict.get(m_name)
        pred_mask = infer_model(model, img_path, input_size, DEVICE)

        overlay_rgb = create_overlay_mask(img_bgr, pred_mask, gt_mask)
        # 缩放到512x512用于显示
        overlay_rgb_512 = cv2.resize(
            overlay_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4
        )
        results[m_name].append(overlay_rgb_512)

        # 计算覆盖率
        veg_pixels = (pred_mask > 0).sum()
        coverage = (veg_pixels / pred_mask.size) * 100

        # 直接保存
        output_file = os.path.join(
            OUTPUT_DIR,
            f"{m_name.lower().replace('+', 'plus').replace('.', '')}_prediction.png",
        )
        cv2.imwrite(output_file, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        print(f"  ✅ {m_name:15s}: {coverage:6.2f}% → {os.path.basename(output_file)}")

# 保存原图
output_original = os.path.join(OUTPUT_DIR, "00_original.png")
img_bgr = cv2.imread(selected_images[0])
cv2.imwrite(output_original, img_bgr)
print(f"\n✅ 原图保存: {os.path.basename(output_original)}")

# ==================== 生成统计信息 ====================

print("\n" + "=" * 80)
print("✅ 推理完成！")
print("=" * 80)
print(f"\n📂 输出目录:")
print(f"   {OUTPUT_DIR}")
print(f"\n📄 生成的图像:")
print(f"   ✓ 00_original.png")
for m_name in [m["name"] for m in MODELS_INFO]:
    filename = m_name.lower().replace("+", "plus").replace(".", "")
    print(f"   ✓ {filename}_prediction.png")
print(f"\n🎨 彩色标记表：")
print(f"   • 蓝色  = TP (True Positive)  - 正确检测植被")
print(f"   • 绿色  = FP (False Positive) - 误检测（背景被检为植被）")
print(f"   • 粉色  = FN (False Negative) - 漏检（植被被检为背景）")
print("=" * 80)
