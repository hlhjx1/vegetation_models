# 紫金山单张图像推理 - Colab 分 Cell 执行指南

**重要提示**：必须在 Colab 中**分别复制粘贴**到不同的 cell 中**按顺序执行**，不要使用 `exec()`！

---

## Cell 1: 导入和修补

```python
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

# 关键：torch.load 全局修补（在所有其他导入之前）
import torch.serialization
_original_load = torch.load

def _patched_load(f, *args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(f, *args, **kwargs)

torch.load = _patched_load
print("✅ torch.load 修补完成")

# 配置路径
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

# 配置 sys.path
if "/content/drive/MyDrive/vegetation_models_v2/2_MobileSAM/code" not in sys.path:
    sys.path.insert(0, "/content/drive/MyDrive/vegetation_models_v2/2_MobileSAM/code")
if "/content/drive/MyDrive/vegetation_models_v2/6_MobileSAMV2/code" not in sys.path:
    sys.path.insert(0, "/content/drive/MyDrive/vegetation_models_v2/6_MobileSAMV2/code")
if "/content/drive/MyDrive/vegetation_models_v2/5_SAM21_Tiny/code" not in sys.path:
    sys.path.insert(0, "/content/drive/MyDrive/vegetation_models_v2/5_SAM21_Tiny/code")

# 改变工作目录
_sam2_code_dir = os.path.join(BASE_DIR, "1_SAM2_Tiny/code")
if os.path.exists(_sam2_code_dir):
    os.chdir(_sam2_code_dir)
    if _sam2_code_dir not in sys.path:
        sys.path.insert(0, _sam2_code_dir)
    print(f"✅ 工作目录已改为: {_sam2_code_dir}")

print("✅ Cell 1 完成！")
```

---

## Cell 2: 定义模型加载函数

```python
print("⏳ 定义模型加载函数...")

def load_unet(base_dir, device):
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
            print(f"✅ UNet加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ UNet加载失败: {e}")
        return None

def load_deeplabv3(base_dir, device):
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
            print(f"✅ DeepLabV3+加载成功")
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
    try:
        import torch.nn as nn
        from mobile_sam import sam_model_registry

        class MobileSAMSeg(nn.Module):
            def __init__(self, ckpt_path, num_classes=2):
                super().__init__()
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

                if os.path.exists(ckpt_path):
                    try:
                        ckpt = torch.load(ckpt_path, map_location=device)
                        if isinstance(ckpt, dict) and "model_state" in ckpt:
                            ckpt_state = ckpt["model_state"]
                        else:
                            ckpt_state = ckpt
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
    try:
        import torch.nn as nn
        from sam2.build_sam import build_sam2, build_sam2_hf

        class SAM21Seg(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                try:
                    sam21 = build_sam2(
                        "sam2.1/sam2.1_hiera_t.yaml",
                        "/content/drive/MyDrive/vegetation_models_v2/5_SAM21_Tiny/weights/sam2.1_hiera_tiny.pt",
                        device=device,
                        mode="train",
                    )
                except:
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
    try:
        import torch.nn as nn
        from mobile_sam import sam_model_registry

        class MobileSAMV2Seg(nn.Module):
            def __init__(self, ckpt_path, num_classes=2):
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

                if os.path.exists(ckpt_path):
                    try:
                        ckpt = torch.load(ckpt_path, map_location=device)
                        if isinstance(ckpt, dict) and "model_state" in ckpt:
                            ckpt_state = ckpt["model_state"]
                        else:
                            ckpt_state = ckpt
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

print("✅ Cell 2 完成！")
```

---

## Cell 3: 加载模型和运行推理

```python
print("=" * 80)
print("🎯 紫金山单张图像推理 - 加载模型")
print("=" * 80)

print(f"\n⏳ 加载所有6个模型...")

models_dict = {
    "UNet": load_unet(BASE_DIR, DEVICE),
    "DeepLabV3+": load_deeplabv3(BASE_DIR, DEVICE),
    "SAM2_Tiny": load_sam2_tiny(BASE_DIR, DEVICE),
    "MobileSAM": load_mobilesam(BASE_DIR, DEVICE),
    "SAM2.1_Tiny": load_sam21_tiny(BASE_DIR, DEVICE),
    "MobileSAMV2": load_mobilesamv2(BASE_DIR, DEVICE),
}

print(f"✅ 所有模型加载完成！\n")

# 定义推理函数
def create_overlay_mask(img_bgr, pred_mask, gt_mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    base_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    tp_mask = (pred_mask == 1) & (gt_mask == 1)
    fp_mask = (pred_mask == 1) & (gt_mask == 0)
    fn_mask = (pred_mask == 0) & (gt_mask == 1)
    color_mask = np.zeros_like(base_img)
    color_mask[tp_mask] = [0, 0, 255]
    color_mask[fp_mask] = [0, 255, 0]
    color_mask[fn_mask] = [255, 0, 255]
    alpha = 0.5
    overlay_img = np.where(
        color_mask != 0,
        cv2.addWeighted(base_img, 1 - alpha, color_mask, alpha, 0),
        base_img,
    )
    return overlay_img

def preprocess_image(img_path, target_size, device):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
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
            preds = cv2.resize(preds.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
        return preds
    except Exception as e:
        print(f"⚠️ 推理失败: {e}")
        return np.zeros((512, 512), dtype=np.uint8)

# 选择第一张图像
all_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
img_path = all_images[0]
filename = os.path.splitext(os.path.basename(img_path))[0]

print(f"✅ 选择第1张图像: {filename}.png")

# 读取图像和标签
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

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

# 保存原图
output_original = os.path.join(OUTPUT_DIR, "01_original_image.png")
Image.fromarray(img_rgb).save(output_original)
print(f"✅ 原图已保存\n")

# 推理
print("⏳ 模型推理...")
results = {}

for m_info in MODELS_INFO:
    m_name = m_info["name"]
    input_size = m_info["size"]
    model = models_dict.get(m_name)
    pred_mask = infer_model(model, img_path, input_size, DEVICE)
    veg_pixels = (pred_mask > 0).sum()
    coverage = (veg_pixels / pred_mask.size) * 100
    results[m_name] = {"mask": pred_mask, "coverage": coverage}
    print(f"  {m_name:15s}: {coverage:6.2f}% 植被覆盖")

# 保存结果
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
        overlay_rgb = create_overlay_mask(img_bgr, pred_mask, gt_mask)
        output_file = os.path.join(OUTPUT_DIR, f"{save_prefix}_prediction.png")
        cv2.imwrite(output_file, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        coverage = results[m_name]["coverage"]
        print(f"  ✅ {m_name:15s} → {save_prefix}_prediction.png ({coverage:.2f}%)")

# 总结
print("\n" + "=" * 80)
print("✅ 推理完成！")
print("=" * 80)
print(f"\n📂 输出目录: {OUTPUT_DIR}")
print(f"\n📊 植被覆盖率统计:")
for m_name, data in results.items():
    print(f"   {m_name:15s}: {data['coverage']:7.2f}%")
print("\n" + "=" * 80)
```

---

## 💡 使用方法

1. **复制 Cell 1 到 Colab 的第1个cell 并运行**
2. **复制 Cell 2 到 Colab 的第2个cell 并运行**（加载函数定义）
3. **复制 Cell 3 到 Colab 的第3个cell 并运行**（执行推理）

**这个方式 100% 有效**，因为它完全模拟了笔记本的执行方式！
