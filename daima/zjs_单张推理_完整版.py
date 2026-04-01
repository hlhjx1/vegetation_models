# ==============================================================================
# 紫金山单张图像推理 - 完整版（完全复制自生成模型用推理图.ipynb的工作代码）
# 改为单张图像而非13个案例
# ==============================================================================

print("=" * 80)
print("🎯 紫金山单张图像推理 - 6模型预测（完整版）")
print("=" * 80)

import os
import sys
import cv2
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import warnings

warnings.filterwarnings("ignore")

# ================================================================================
# 关键步骤1：全局修补 torch.load（必须在所有导入之后立即做）
# ================================================================================
import torch.serialization

_original_load = torch.load


def _patched_load(f, *args, **kwargs):
    """允许加载包含numpy/复杂类型的权重"""
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(f, *args, **kwargs)


torch.load = _patched_load
print("✅ torch.load 全局修补完成")

# ================================================================================
# 关键步骤1.5：在修补后立即导入所有关键库（防止导入时的recursion）
# ================================================================================
print("⏳ 预加载关键库...")
try:
    import segmentation_models_pytorch as smp

    print("✅ segmentation_models_pytorch 预加载完成")
except:
    print("⚠️ segmentation_models_pytorch 预加载失败，延迟导入")
    smp = None

# ================================================================================
# 配置路径
# ================================================================================

BASE_DIR = "/content/drive/MyDrive/vegetation_models_v2"
DATASET_DIR = "/content/drive/MyDrive/datasets/2024-seg"
IMAGE_DIR = os.path.join(DATASET_DIR, "JPEGImages")
MASK_DIR = os.path.join(DATASET_DIR, "SegmentationClass")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_zjs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 当前设备: {DEVICE}")

MODELS_INFO = [
    {"name": "UNet", "size": 512},
    {"name": "DeepLabV3+", "size": 512},
    {"name": "SAM2_Tiny", "size": 1024},
    {"name": "MobileSAM", "size": 1024},
    {"name": "SAM2.1_Tiny", "size": 1024},
    {"name": "MobileSAMV2", "size": 1024},
]

# ================================================================================
# 关键步骤2：配置 sys.path 和工作目录
# ================================================================================

print(f"\n⏳ 配置模型代码路径...")

if "/content/drive/MyDrive/vegetation_models_v2/2_MobileSAM/code" not in sys.path:
    sys.path.insert(0, "/content/drive/MyDrive/vegetation_models_v2/2_MobileSAM/code")

if "/content/drive/MyDrive/vegetation_models_v2/6_MobileSAMV2/code" not in sys.path:
    sys.path.insert(0, "/content/drive/MyDrive/vegetation_models_v2/6_MobileSAMV2/code")

if "/content/drive/MyDrive/vegetation_models_v2/5_SAM21_Tiny/code" not in sys.path:
    sys.path.insert(0, "/content/drive/MyDrive/vegetation_models_v2/5_SAM21_Tiny/code")

# 改变工作目录使 Hydra 能找到 config
_original_cwd = os.getcwd()
_sam2_code_dir = os.path.join(BASE_DIR, "1_SAM2_Tiny/code")
if os.path.exists(_sam2_code_dir):
    os.chdir(_sam2_code_dir)
    if _sam2_code_dir not in sys.path:
        sys.path.insert(0, _sam2_code_dir)
    print(f"✅ 工作目录已改为: {_sam2_code_dir}")

print("✅ 路径配置完成\n")

# ================================================================================
# 模型加载函数 - 完全复制自工作笔记本
# ================================================================================

print("⏳ 定义模型加载函数...")


def load_unet(base_dir, device):
    """UNet"""
    try:
        # 重新确保torch.load已修补
        import torch.serialization

        if not hasattr(torch.load, "_is_patched"):
            _original_load = (
                torch.load.__wrapped__
                if hasattr(torch.load, "__wrapped__")
                else torch.serialization.load
            )

            def _ensure_patched(f, *args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return _original_load(f, *args, **kwargs)

            _ensure_patched._is_patched = True
            torch.load = _ensure_patched

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
            print(f"✅ UNet加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ UNet加载失败: {e}")
        return None


def load_deeplabv3(base_dir, device):
    """DeepLabV3+"""
    try:
        # 重新确保torch.load已修补
        import torch.serialization

        if not hasattr(torch.load, "_is_patched"):
            _original_load = (
                torch.load.__wrapped__
                if hasattr(torch.load, "__wrapped__")
                else torch.serialization.load
            )

            def _ensure_patched(f, *args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return _original_load(f, *args, **kwargs)

            _ensure_patched._is_patched = True
            torch.load = _ensure_patched

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
            print(f"✅ DeepLabV3+加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ DeepLabV3+加载失败: {e}")
        return None


def load_sam2_tiny(base_dir, device):
    """SAM2-Tiny"""
    try:
        import torch.nn as nn
        from sam2.build_sam import build_sam2

        class SAM2Seg(nn.Module):
            def __init__(self, cfg_name, ckpt_path, num_classes=2):
                super().__init__()
                # Hydra会在code目录中找到sam2/sam2_hiera_t.yaml
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
                feat = (
                    features["backbone_fpn"][-1]
                    if isinstance(features, dict)
                    else features
                )
                return self.seg_head(feat)

        ckpt = os.path.join(base_dir, "1_SAM2_Tiny/weights/sam2_hiera_tiny.pt")
        model = SAM2Seg("sam2/sam2_hiera_t.yaml", ckpt, num_classes=2).to(device)

        best_ckpt = os.path.join(base_dir, "1_SAM2_Tiny/checkpoints/sam2tiny_best.pth")
        if os.path.exists(best_ckpt):
            try:
                ckpt_state = torch.load(best_ckpt, map_location=device)
                model.load_state_dict(
                    ckpt_state.get("model_state", ckpt_state), strict=False
                )
            except:
                pass

        print(f"✅ SAM2_Tiny加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ SAM2_Tiny加载失败: {e}")
        return None


def load_mobilesam(base_dir, device):
    """MobileSAM"""
    try:
        import torch.nn as nn
        from mobile_sam import sam_model_registry

        class MobileSAMSeg(nn.Module):
            def __init__(self, ckpt_path, num_classes=2):
                super().__init__()
                # 先从pretrain weights初始化
                pretrain = os.path.join(base_dir, "2_MobileSAM/weights/mobile_sam.pt")
                sam = sam_model_registry["vit_t"](
                    checkpoint=pretrain if os.path.exists(pretrain) else None
                )
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

                # 加载checkpoint中的seg_head权重（如果存在）
                if os.path.exists(ckpt_path):
                    try:
                        ckpt = torch.load(ckpt_path, map_location=device)
                        if isinstance(ckpt, dict) and "model_state" in ckpt:
                            ckpt_state = ckpt["model_state"]
                        else:
                            ckpt_state = ckpt
                        # 只加载seg_head的权重，忽略其他不匹配的部分
                        seg_head_state = {
                            k.replace("seg_head.", ""): v
                            for k, v in ckpt_state.items()
                            if "seg_head" in k
                        }
                        if seg_head_state:
                            self.seg_head.load_state_dict(seg_head_state, strict=False)
                    except:
                        pass

            def forward(self, x):
                return self.seg_head(self.encoder(x))

        ckpt_path = os.path.join(base_dir, "2_MobileSAM/checkpoints/mobilesam_best.pth")
        model = MobileSAMSeg(ckpt_path, num_classes=2).to(device)
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
            def __init__(self, num_classes=2):
                super().__init__()
                # 优先用本地config sam2.1/sam2.1_hiera_t.yaml
                try:
                    sam21 = build_sam2(
                        "sam2.1/sam2.1_hiera_t.yaml",
                        "/content/drive/MyDrive/vegetation_models_v2/5_SAM21_Tiny/weights/sam2.1_hiera_tiny.pt",
                        device=device,
                        mode="train",
                    )
                except:
                    # 后备方案：用hub版本
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
                feat = (
                    features["backbone_fpn"][-1]
                    if isinstance(features, dict)
                    else features
                )
                return self.seg_head(feat)

        model = SAM21Seg(num_classes=2).to(device)

        best_ckpt = os.path.join(
            base_dir, "5_SAM21_Tiny/checkpoints/sam21tiny_best.pth"
        )
        if os.path.exists(best_ckpt):
            try:
                ckpt_state = torch.load(best_ckpt, map_location=device)
                model.load_state_dict(
                    ckpt_state.get("model_state", ckpt_state), strict=False
                )
            except:
                pass

        print(f"✅ SAM2.1_Tiny加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ SAM2.1_Tiny加载失败: {e}")
        return None


def load_mobilesamv2(base_dir, device):
    """MobileSAMV2"""
    try:
        import torch.nn as nn
        from mobile_sam import sam_model_registry

        class MobileSAMV2Seg(nn.Module):
            def __init__(self, ckpt_path, num_classes=2):
                super().__init__()
                # 有pretrain weights就用，没有就直接初始化
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

                # 加载checkpoint
                if os.path.exists(ckpt_path):
                    try:
                        ckpt = torch.load(ckpt_path, map_location=device)
                        if isinstance(ckpt, dict) and "model_state" in ckpt:
                            ckpt_state = ckpt["model_state"]
                        else:
                            ckpt_state = ckpt
                        # 提取seg_head权重
                        seg_head_state = {
                            k.replace("seg_head.", ""): v
                            for k, v in ckpt_state.items()
                            if "seg_head" in k
                        }
                        if seg_head_state:
                            self.seg_head.load_state_dict(seg_head_state, strict=False)
                    except:
                        pass

            def forward(self, x):
                return self.seg_head(self.encoder(x))

        ckpt_path = os.path.join(
            base_dir, "6_MobileSAMV2/checkpoints/mobilesamv2_best.pth"
        )
        model = MobileSAMV2Seg(ckpt_path, num_classes=2).to(device)
        print(f"✅ MobileSAMV2加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ MobileSAMV2加载失败: {e}")
        return None


print("✅ 所有函数定义完成\n")

# ================================================================================
# 加载所有模型
# ================================================================================

print("⏳ 加载所有6个模型...")

models_dict = {
    "UNet": load_unet(BASE_DIR, DEVICE),
    "DeepLabV3+": load_deeplabv3(BASE_DIR, DEVICE),
    "SAM2_Tiny": load_sam2_tiny(BASE_DIR, DEVICE),
    "MobileSAM": load_mobilesam(BASE_DIR, DEVICE),
    "SAM2.1_Tiny": load_sam21_tiny(BASE_DIR, DEVICE),
    "MobileSAMV2": load_mobilesamv2(BASE_DIR, DEVICE),
}

print("✅ 所有模型加载完成\n")

# ================================================================================
# 推理函数 - 完全复制自工作笔记本
# ================================================================================


def create_overlay_mask(img_bgr, pred_mask, gt_mask):
    """生成彩色对比图"""
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
    """图像预处理"""
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
    """使用模型进行推理"""
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


# ================================================================================
# 执行单张图像推理
# ================================================================================

print("⏳ 执行推理...\n")

# 选择第一张图像
all_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))

if not all_images:
    print("❌ 未找到图像")
else:
    img_path = all_images[0]
    filename = os.path.splitext(os.path.basename(img_path))[0]

    print(f"✅ 选择第1张图像: {filename}.png")

    # 读取图像
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 读取Ground Truth标签
    mask_path = os.path.join(MASK_DIR, filename + ".png")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(MASK_DIR, filename + ".jpg")

    if os.path.exists(mask_path):
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        gt_mask = (gt_mask > 0).astype(np.uint8)
        print(f"✅ Ground Truth标签已加载")
    else:
        gt_mask = np.zeros((512, 512), dtype=np.uint8)
        print(f"⚠️ Ground Truth标签不存在")

    # 保存原始图像
    output_original = os.path.join(OUTPUT_DIR, "01_original_image.png")
    Image.fromarray(img_rgb).save(output_original)
    print(f"✅ 原图已保存: {output_original}\n")

    # 执行推理
    print("⏳ 模型推理...")
    results = {}

    for m_info in MODELS_INFO:
        m_name = m_info["name"]
        input_size = m_info["size"]

        model = models_dict.get(m_name)
        pred_mask = infer_model(model, img_path, input_size, DEVICE)

        # 计算植被覆盖率
        veg_pixels = (pred_mask > 0).sum()
        coverage = (veg_pixels / pred_mask.size) * 100

        results[m_name] = {"mask": pred_mask, "coverage": coverage}
        print(f"  {m_name:15s}: {coverage:6.2f}% 植被覆盖")

    # 保存预测结果
    print(f"\n⏳ 保存预测结果...")

    model_name_map = {
        "UNet": "02_unet",
        "DeepLabV3+": "03_deeplabv3plus",
        "SAM2_Tiny": "04_sam2_tiny",
        "MobileSAM": "05_mobilesam",
        "SAM2.1_Tiny": "06_sam21_tiny",
        "MobileSAMV2": "07_mobilesamv2",
    }

    for m_name, save_prefix in model_name_map.items():
        if m_name in results:
            pred_mask = results[m_name]["mask"]

            # 生成彩色对比图
            overlay_rgb = create_overlay_mask(img_bgr, pred_mask, gt_mask)

            # 保存
            output_file = os.path.join(OUTPUT_DIR, f"{save_prefix}_prediction.png")
            cv2.imwrite(output_file, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            coverage = results[m_name]["coverage"]
            print(f"  ✅ {m_name:15s} → {save_prefix}_prediction.png ({coverage:.2f}%)")

    # 输出总结
    print("\n" + "=" * 80)
    print("✅ 推理完成！")
    print("=" * 80)
    print(f"\n📂 输出目录: {OUTPUT_DIR}")
    print(f"\n📄 生成的7张图像:")
    print(f"   ✓ 01_original_image.png          （原始RGB图像）")
    print(f"   ✓ 02_unet_prediction.png         （UNet预测 - 彩色对比）")
    print(f"   ✓ 03_deeplabv3plus_prediction.png（DeepLabV3+预测 - 彩色对比）")
    print(f"   ✓ 04_sam2_tiny_prediction.png    （SAM2_Tiny预测 - 彩色对比）")
    print(f"   ✓ 05_mobilesam_prediction.png    （MobileSAM预测 - 彩色对比）")
    print(f"   ✓ 06_sam21_tiny_prediction.png   （SAM2.1_Tiny预测 - 彩色对比）")
    print(f"   ✓ 07_mobilesamv2_prediction.png  （MobileSAMV2预测 - 彩色对比）")

    print(f"\n📊 植被覆盖率统计:")
    for m_name, data in results.items():
        print(f"   {m_name:15s}: {data['coverage']:7.2f}%")

    print(f"\n🎨 彩色标记表：")
    print(f"   • 蓝色  = TP (True Positive)  - 正确检测植被")
    print(f"   • 绿色  = FP (False Positive) - 误检测（背景被检为植被）")
    print(f"   • 粉色  = FN (False Negative) - 漏检（植被被检为背景）")
    print("=" * 80)
