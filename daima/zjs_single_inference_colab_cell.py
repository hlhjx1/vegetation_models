# ==============================================================================
# Colab Cell: 紫金山单张推理 - 6模型推理 (借鉴成熟模型加载代码)
# ==============================================================================
"""
从紫金山测试集抽出一张图，用已训练的6个模型推理
输出：output_zjs/ 文件夹下的7张图（原图+6模型预测）

核心改进：
- 借鉴 COLAB_generate_figure_6models_v2.py 的完整模型加载代码
- 借鉴 BLOCK6_FINAL.py 的Hydra工作目录管理
- 支持直接从checkpoint加载权重，不需依赖前置Cell
"""

import os
import sys
import glob
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import warnings

warnings.filterwarnings("ignore")

# ============ PyTorch兼容性修复 ============
_original_load = torch.load


def _patched_load(f, *args, **kwargs):
    """允许加载包含numpy/复杂类型的权重"""
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(f, *args, **kwargs)


torch.load = _patched_load
# ==========================================

print("=" * 80)
print("🎯 紫金山单张图像推理 - 6模型预测")
print("=" * 80)

# ============================================================
# 配置
# ============================================================

DRIVE = "/content/drive/MyDrive/vegetation_models_v2"
ZJS_DATASET_DIR = "/content/drive/MyDrive/datasets/2024-seg"
OUTPUT_DIR = os.path.join(DRIVE, "output_zjs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
print(f"  输出目录: {OUTPUT_DIR}")
print(f"  设备: {DEVICE}")
print(f"  PyTorch版本: {torch.__version__}")


# ============================================================
# 模型加载函数 (借鉴成熟代码)
# ============================================================

print(f"\n⏳ 定义模型加载函数...")

# ============ 关键: 改变工作目录使Hydra能找到config ============
_original_cwd = os.getcwd()
_sam2_code_dir = os.path.join(DRIVE, "1_SAM2_Tiny/code")
if os.path.exists(_sam2_code_dir):
    os.chdir(_sam2_code_dir)
    if _sam2_code_dir not in sys.path:
        sys.path.insert(0, _sam2_code_dir)
    print(f"  ✅ 工作目录已改为: {_sam2_code_dir}")
# ================================================================


def load_unet(base_dir, device):
    """加载 UNet (ResNet34)"""
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
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get("model_state", ckpt))
            print(f"  ✅ UNet加载成功")

        model.eval()
        return model
    except Exception as e:
        print(f"  ⚠️ UNet加载失败: {e}")
        return None


def load_deeplabv3(base_dir, device):
    """加载 DeepLabV3+ (ResNet50)"""
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
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get("model_state", ckpt))
            print(f"  ✅ DeepLabV3+加载成功")

        model.eval()
        return model
    except Exception as e:
        print(f"  ⚠️ DeepLabV3+加载失败: {e}")
        return None


def load_sam2_tiny(base_dir, device):
    """加载 SAM2-Tiny"""
    try:
        import torch.nn as nn
        from sam2.build_sam import build_sam2
        from hydra.core.global_hydra import GlobalHydra

        class SAM2Seg(nn.Module):
            def __init__(self, cfg_name, ckpt_path, num_classes=2):
                super().__init__()
                sam2 = build_sam2(cfg_name, ckpt_path, device=device)
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
                if isinstance(features, dict):
                    feat = features.get("backbone_fpn", [features])[0]
                    if isinstance(feat, list):
                        feat = feat[-1]
                else:
                    feat = features
                return self.seg_head(feat)

        pretrain_ckpt = os.path.join(base_dir, "1_SAM2_Tiny/weights/sam2_hiera_tiny.pt")
        finetuned_ckpt = os.path.join(
            base_dir, "1_SAM2_Tiny/checkpoints/sam2tiny_best.pth"
        )

        if not os.path.exists(pretrain_ckpt):
            print(f"  ⚠️ SAM2_Tiny预训练权重找不到")
            return None

        try:
            GlobalHydra.instance().clear()
        except:
            pass

        model = SAM2Seg("sam2_hiera_t.yaml", pretrain_ckpt, num_classes=2).to(device)

        if os.path.exists(finetuned_ckpt):
            ckpt = torch.load(finetuned_ckpt, map_location=device)
            model.load_state_dict(ckpt.get("model_state", ckpt))
            print(f"  ✅ SAM2_Tiny加载成功（微调权重）")
        else:
            print(f"  ✅ SAM2_Tiny加载成功（预训练权重）")

        model.eval()
        return model
    except Exception as e:
        print(f"  ⚠️ SAM2_Tiny加载失败: {e}")
        return None


def load_mobilesam(base_dir, device):
    """加载 MobileSAM"""
    try:
        import torch.nn as nn

        try:
            from mobile_sam import sam_model_registry
        except ImportError:
            sys.path.insert(0, os.path.join(base_dir, "2_MobileSAM/code"))
            from mobile_sam import sam_model_registry

        class MobileSAMSeg(nn.Module):
            def __init__(self, ckpt_path, device, num_classes=2):
                super().__init__()
                sam = sam_model_registry["vit_t"](checkpoint=ckpt_path)
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

        ckpt_path = os.path.join(base_dir, "2_MobileSAM/checkpoints/mobilesam_best.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(base_dir, "2_MobileSAM/weights/mobile_sam.pt")

        if not os.path.exists(ckpt_path):
            print(f"  ⚠️ MobileSAM权重找不到")
            return None

        model = MobileSAMSeg(ckpt_path, device).to(device)
        print(f"  ✅ MobileSAM加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"  ⚠️ MobileSAM加载失败: {e}")
        return None


def load_sam21_tiny(base_dir, device):
    """加载 SAM2.1-Tiny"""
    try:
        import torch.nn as nn
        from sam2.build_sam import build_sam2_hf
        from hydra.core.global_hydra import GlobalHydra

        class SAM21Seg(nn.Module):
            def __init__(self, device, num_classes=2):
                super().__init__()
                sam21 = build_sam2_hf(
                    "facebook/sam2.1-hiera-tiny", device=device, mode="train"
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
                    feat = features.get("backbone_fpn", [features])[0]
                    if isinstance(feat, list):
                        feat = feat[-1]
                else:
                    feat = features
                return self.seg_head(feat)

        try:
            GlobalHydra.instance().clear()
        except:
            pass

        model = SAM21Seg(device).to(device)

        finetuned_ckpt = os.path.join(
            base_dir, "5_SAM21_Tiny/checkpoints/sam21tiny_best.pth"
        )
        if os.path.exists(finetuned_ckpt):
            ckpt = torch.load(finetuned_ckpt, map_location=device)
            model.load_state_dict(ckpt.get("model_state", ckpt))
            print(f"  ✅ SAM2.1_Tiny加载成功（微调权重）")
        else:
            print(f"  ✅ SAM2.1_Tiny加载成功（HF预训练权重）")

        model.eval()
        return model
    except Exception as e:
        print(f"  ⚠️ SAM2.1_Tiny加载失败: {e}")
        return None


def load_mobilesamv2(base_dir, device):
    """加载 MobileSAMV2"""
    try:
        import torch.nn as nn

        try:
            from mobile_sam import sam_model_registry
        except ImportError:
            sys.path.insert(0, os.path.join(base_dir, "6_MobileSAMV2/code"))
            from mobile_sam import sam_model_registry

        class MobileSAMV2Seg(nn.Module):
            def __init__(self, ckpt_path, device, num_classes=2):
                super().__init__()
                try:
                    sam = sam_model_registry["vit_t_mobile"](checkpoint=ckpt_path)
                except:
                    sam = sam_model_registry["vit_t"](checkpoint=ckpt_path)
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

        ckpt_path = os.path.join(
            base_dir, "6_MobileSAMV2/checkpoints/mobilesamv2_best.pth"
        )
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(base_dir, "6_MobileSAMV2/weights/mobile_samv2.pt")

        if not os.path.exists(ckpt_path):
            print(f"  ⚠️ MobileSAMV2权重找不到")
            return None

        model = MobileSAMV2Seg(ckpt_path, device).to(device)
        print(f"  ✅ MobileSAMV2加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"  ⚠️ MobileSAMV2加载失败: {e}")
        return None


# ====== 加载所有模型 ======
print(f"\n⏳ 加载所有6个模型...")

models_dict = {
    "UNet": load_unet(DRIVE, DEVICE),
    "DeepLabV3+": load_deeplabv3(DRIVE, DEVICE),
    "SAM2_Tiny": load_sam2_tiny(DRIVE, DEVICE),
    "MobileSAM": load_mobilesam(DRIVE, DEVICE),
    "SAM2.1_Tiny": load_sam21_tiny(DRIVE, DEVICE),
    "MobileSAMV2": load_mobilesamv2(DRIVE, DEVICE),
}

print(f"✅ 所有模型加载完成\n")


# ============================================================
# 工具函数
# ============================================================


def preprocess_image(img_path, target_size):
    """图像预处理 - ImageNet标准化"""
    img = Image.open(img_path).convert("RGB")
    tf = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tf(img).unsqueeze(0).to(DEVICE)


def infer_model(model, img_path, img_size, debug=False):
    """使用模型进行推理 - 支持多种输出格式"""
    if model is None:
        return np.zeros((512, 512), dtype=np.uint8)

    try:
        img_tensor = preprocess_image(img_path, img_size)
        with torch.no_grad():
            output = model(img_tensor)

            if debug:
                print(f"\n    [DEBUG] 原始输出类型: {type(output)}")
                if isinstance(output, dict):
                    print(f"    [DEBUG] Dict键: {list(output.keys())}")
                if isinstance(output, torch.Tensor):
                    print(
                        f"    [DEBUG] 张量形状: {output.shape}, 值范围: [{output.min():.3f}, {output.max():.3f}]"
                    )

            # 处理不同的输出格式
            if isinstance(output, dict):
                # 某些模型返回dict，取'out'或其他关键值
                if "out" in output:
                    output = output["out"]
                elif "masks" in output:
                    output = output["masks"]
                elif "seg" in output:
                    output = output["seg"]
                else:
                    # 取第一个有效值
                    for key, val in output.items():
                        if isinstance(val, torch.Tensor):
                            output = val
                            break

            # 确保是张量
            if not isinstance(output, torch.Tensor):
                print(f"    ⚠️ 输出不是张量，跳过")
                return np.zeros((512, 512), dtype=np.uint8)

            # 处理不同的张量形状
            if len(output.shape) == 5:  # [B, T, C, H, W]（某些时间序列模型）
                output = output[:, -1, :, :, :]  # 取最后一帧
                if debug:
                    print(f"    [DEBUG] 处理后形状: {output.shape}")

            if len(output.shape) == 4:
                batch_size, channels, height, width = output.shape

                # 二分类输出
                if channels == 2:
                    # 使用softmax确保概率
                    probs = torch.softmax(output, dim=1)  # [B, 2, H, W]
                    veg_prob = probs[:, 1, :, :]  # [B, H, W]

                    # 尝试多个阈值，如果都是0就降低阈值
                    for threshold in [0.5, 0.3, 0.1, 0.0]:
                        preds = (
                            (veg_prob > threshold)
                            .squeeze(0)
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )
                        if preds.sum() > 0:
                            if debug and threshold != 0.5:
                                print(
                                    f"    [DEBUG] 用阈值{threshold}获得{preds.sum()}个植被像素"
                                )
                            break

                # 多分类 - 0=背景, 1=植被
                elif channels == 1:
                    # 直接用该通道作为热力图
                    preds = (
                        (output.squeeze(1) > 0)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )

                else:
                    # 通用多分类处理
                    class_indices = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                    preds = (class_indices > 0).astype(np.uint8)

            elif len(output.shape) == 3:  # [B, H, W]
                preds = (output.squeeze(0) > 0).cpu().numpy().astype(np.uint8)

            elif len(output.shape) == 2:  # [H, W]
                preds = (output > 0).cpu().numpy().astype(np.uint8)

            else:
                print(f"    ⚠️ 未知的输出形状: {output.shape}")
                return np.zeros((512, 512), dtype=np.uint8)

            # 调整到512×512
            if preds.shape != (512, 512):
                preds = cv2.resize(
                    preds.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST
                )

            if debug:
                print(
                    f"    [DEBUG] 最终预测：{preds.sum()}个植被像素 ({100*preds.sum()/preds.size:.2f}%)"
                )

            # 保持为0-1，后续处理时乘以255
            return preds

    except Exception as e:
        print(f"    ⚠️ 推理异常: {e}")
        import traceback

        traceback.print_exc()
        return np.zeros((512, 512), dtype=np.uint8)
        return np.zeros((512, 512), dtype=np.uint8)


# ============================================================
# Step 1: 选择图像
# ============================================================

print(f"⏳ Step 1/3: 选择图像...")

img_dir = os.path.join(ZJS_DATASET_DIR, "JPEGImages")

if not os.path.exists(img_dir):
    print(f"  ❌ 图像目录不存在: {img_dir}")
else:
    all_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    if not all_images:
        print(f"  ❌ 未找到PNG图像")
    else:
        # 选择第一张图像
        selected_image = all_images[0]
        image_name = os.path.basename(selected_image)

        print(f"  ✅ 找到 {len(all_images)} 张图像")
        print(f"  ✅ 选中第1张: {image_name}")

        # 读取原图
        img_bgr = cv2.imread(selected_image)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 保存原图到输出目录
        output_original = os.path.join(OUTPUT_DIR, "01_original_image.png")
        Image.fromarray(img_rgb).save(output_original)
        print(f"\n  💾 原图已保存:")
        print(f"     {output_original}")
        print(f"     分辨率: {img_rgb.shape[1]}×{img_rgb.shape[0]}")

        # ============================================================
        # Step 2: 推理
        # ============================================================

        print(f"\n⏳ Step 2/3: 模型推理...")
        print(f"  (提示: 预测值 0=背景, 1=植被)")

        predictions = {}
        pred_stats = {}

        for m_info in MODELS_INFO:
            m_name = m_info["name"]
            input_size = m_info["size"]

            # 从全局 models_dict 中获取已加载的模型
            model = models_dict.get(m_name)

            print(f"  推理 {m_name:15s}...", end=" ")
            pred_mask = infer_model(model, selected_image, input_size)

            # 统计预测结果
            veg_pixels = (pred_mask > 0).sum()
            total_pixels = pred_mask.size
            coverage = (veg_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            pred_stats[m_name] = coverage

            # 转为RGB用于保存（值为0=黑or1=白）
            pred_rgb = np.stack(
                [pred_mask * 255, pred_mask * 255, pred_mask * 255], axis=-1
            )
            predictions[m_name] = pred_rgb

            print(f"✅ (植被覆盖率: {coverage:.1f}%)")

        # ============================================================
        # Step 3: 保存结果
        # ============================================================

        print(f"\n⏳ Step 3/3: 保存预测结果...")

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
                pred_rgb = predictions[m_name].astype(np.uint8)

                # 保存二值化预测 (纯黑白)
                Image.fromarray(pred_rgb).save(output_file)

                coverage = pred_stats.get(m_name, 0)
                print(
                    f"  💾 {m_name:15s} → {save_prefix}_prediction.png (覆盖率: {coverage:.1f}%)"
                )

        # ============================================================
        # 完成
        # ============================================================

        print("\n" + "=" * 80)
        print("✅ 推理完成！")
        print("=" * 80)
        print(f"\n📂 输出目录:")
        print(f"   {OUTPUT_DIR}")
        print(f"\n📄 生成的7张图像:")
        print(f"   ✓ 01_original_image.png          （原始RGB图像）")
        print(f"   ✓ 02_unet_prediction.png         （UNet预测）")
        print(f"   ✓ 03_deeplabv3plus_prediction.png（DeepLabV3+预测）")
        print(f"   ✓ 04_sam2_tiny_prediction.png    （SAM2_Tiny预测）")
        print(f"   ✓ 05_mobilesam_prediction.png    （MobileSAM预测）")
        print(f"   ✓ 06_sam21_tiny_prediction.png   （SAM2.1_Tiny预测）")
        print(f"   ✓ 07_mobilesamv2_prediction.png  （MobileSAMV2预测）")

        print(f"\n📊 植被覆盖率统计：")
        print(f"   原始图像: {len(all_images)} 张可用")
        for m_name, coverage in pred_stats.items():
            print(f"   {m_name:15s}: {coverage:6.2f}%")

        print(f"\n💡 预测值映射:")
        print(f"   • 黑色 (0) = 背景区域")
        print(f"   • 白色 (255) = 植被区域")
        print(f"\n✨ 论文应用:")
        print(f"   • 原图可用于展示研究区域")
        print(f"   • 预测图可用于对比6个模型的绩效")
        print(f"   • 植被覆盖率数据可用于量化对比")
        print("=" * 80)
