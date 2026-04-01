"""
COLAB版本：高清6模型对比图生成 - 改进版 (带更大的标签文字)

使用方法：
1. 打开 Google Colab
2. 在 Drive 中获取此脚本  
3. 把以下每个代码块复制到单独的Colab cell中，按顺序运行
4. 等待推理完成，获得高清对比图

Dataset要求：
- 数据集目录: /content/drive/MyDrive/datasets/2024-seg/
  ├── JPEGImages/  (RGB图像)
  └── SegmentationClass/  (分割掩码)
- 项目目录: /content/drive/MyDrive/vegetation_models_v2/
"""

# ==============================================================================
# BLOCK 1: 安装所有软件包依赖
# 将此段代码粘贴到Colab第1个cell，运行
# ==============================================================================

print("⏳ Step 1/10: 安装Python软件包...")
!pip install -q segmentation-models-pytorch torch torchvision opencv-python-headless
!pip install -q Pillow matplotlib scikit-learn albumentations hydra-core iopath
!apt-get update -qq && apt-get install -y -qq fonts-dejavu > /dev/null 2>&1

print("✅ 软件包安装完成")


# ==============================================================================
# BLOCK 2: 挂载Google Drive
# 将此段代码粘贴到Colab第2个cell，运行，然后点击链接授权
# ==============================================================================

print("⏳ Step 2/10: 挂载Google Drive...")
from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)
os.chdir('/content/drive/MyDrive')

print("✅ Drive已挂载")
print("当前目录:", os.getcwd())
print()


# ==============================================================================
# BLOCK 3: 导入所有依赖库
# 将此段代码粘贴到Colab第3个cell，运行
# ==============================================================================

print("⏳ Step 3/10: 导入依赖库...")

import sys
import cv2
import glob
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import warnings
import pickle

warnings.filterwarnings('ignore')

# ==================== 关键全局处理：解决numpy/torch不兼容 ====================
# 禁用torch的严格权重检查（支持旧模型和包含numpy类型的权重文件）
import torch.serialization
_original_load = torch.load

def _patched_load(f, *args, **kwargs):
    """允许加载包含numpy/复杂类型的权重"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)

torch.load = _patched_load
# ========================================================================

print(f"✅ PyTorch版本: {torch.__version__}")
print(f"✅ GPU可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU型号: {torch.cuda.get_device_name()}")
print()


# ==============================================================================
# BLOCK 4: 配置路径和参数
# 将此段代码粘贴到Colab第4个cell，运行
# ==============================================================================

print("⏳ Step 4/10: 配置参数...")

BASE_DIR = "/content/drive/MyDrive/vegetation_models_v2"
DATASET_DIR = "/content/drive/MyDrive/datasets/2024-seg"
IMAGE_DIR = os.path.join(DATASET_DIR, "JPEGImages")
MASK_DIR = os.path.join(DATASET_DIR, "SegmentationClass")
OUTPUT_DIR = os.path.join(BASE_DIR, "paper_figures")

NUM_CASES = 13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 6个模型 (不含SegEarth_OV和YOLO11)
MODELS_INFO = [
    {"name": "UNet", "key": "7_UNet", "size": 512, "ckpt": "unet_best.pth"},
    {"name": "DeepLabV3+", "key": "8_DeepLabV3", "size": 512, "ckpt": "deeplabv3plus_best.pth"},
    {"name": "SAM2_Tiny", "key": "1_SAM2_Tiny", "size": 1024, "ckpt": "sam2tiny_best.pth"},
    {"name": "MobileSAM", "key": "2_MobileSAM", "size": 1024, "ckpt": "mobilesam_best.pth"},
    {"name": "SAM2.1_Tiny", "key": "5_SAM21_Tiny", "size": 1024, "ckpt": "sam21tiny_best.pth"},
    {"name": "MobileSAMV2", "key": "6_MobileSAMV2", "size": 1024, "ckpt": "mobilesamv2_best.pth"},
]

print(f"✅ 基础目录: {BASE_DIR}")
print(f"✅ 数据集目录: {DATASET_DIR}")
print(f"✅ 输出目录: {OUTPUT_DIR}")

# 检查数据集是否存在
if os.path.exists(IMAGE_DIR):
    all_images = sorted(
        glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
        glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    )
    print(f"✅ 找到 {len(all_images)} 张图像")
else:
    print(f"❌ 错误: 图像目录不存在 {IMAGE_DIR}")

print()


# ==============================================================================
# BLOCK 5: 安装模型代码（可能需要5-10分钟）
# 将此段代码粘贴到Colab第5个cell，运行
# ==============================================================================

print("⏳ Step 5/10: 安装模型代码...")

!pip install -q -e /content/drive/MyDrive/vegetation_models_v2/1_SAM2_Tiny/code
!pip install -q -e /content/drive/MyDrive/vegetation_models_v2/2_MobileSAM/code
!pip install -q -e /content/drive/MyDrive/vegetation_models_v2/6_MobileSAMV2/code

# 如果上面的安装失败，尝试直接sys.path
if '/content/drive/MyDrive/vegetation_models_v2/2_MobileSAM/code' not in sys.path:
    sys.path.insert(0, '/content/drive/MyDrive/vegetation_models_v2/2_MobileSAM/code')

if '/content/drive/MyDrive/vegetation_models_v2/6_MobileSAMV2/code' not in sys.path:
    sys.path.insert(0, '/content/drive/MyDrive/vegetation_models_v2/6_MobileSAMV2/code')

# SAM2.1手动添加到环境
if '/content/drive/MyDrive/vegetation_models_v2/5_SAM21_Tiny/code' not in sys.path:
    sys.path.insert(0, '/content/drive/MyDrive/vegetation_models_v2/5_SAM21_Tiny/code')

print("✅ 模型代码安装完成")
print()


# ==============================================================================
# BLOCK 6: 定义模型加载函数
# 将此段代码粘贴到Colab第6个cell，运行
# ==============================================================================

print("⏳ Step 6/10: 定义模型加载函数...")

def load_unet(base_dir, device):
    try:
        import segmentation_models_pytorch as smp
        
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=2,
            activation=None
        ).to(device)
        
        ckpt_path = os.path.join(base_dir, "7_UNet/checkpoints/unet_best.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
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
    try:
        import segmentation_models_pytorch as smp
        
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=2,
            encoder_output_stride=16,
            activation=None
        ).to(device)
        
        ckpt_path = os.path.join(base_dir, "8_DeepLabV3/checkpoints/deeplabv3plus_best.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
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
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        
        class SAM2Seg(nn.Module):
            def __init__(self, cfg, ckpt_path, device, num_classes=2):
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
                    nn.Upsample(size=(1024, 1024), mode='bilinear', align_corners=False)
                )
            
            def forward(self, x):
                features = self.encoder(x)
                # SAM2 encoder返回dict，取backbone_fpn最后一层
                if isinstance(features, dict):
                    feat = features['backbone_fpn'][-1]
                else:
                    feat = features
                return self.seg_head(feat)
        
        # Pretrained权重路径
        pretrain_ckpt = os.path.join(base_dir, "1_SAM2_Tiny/weights/sam2_hiera_tiny.pt")
        if not os.path.exists(pretrain_ckpt):
            print(f"⚠️ SAM2_Tiny预训练权重找不到: {pretrain_ckpt}")
            return None
        
        # 微调后的权重路径
        finetuned_ckpt = os.path.join(base_dir, "1_SAM2_Tiny/checkpoints/sam2tiny_best.pth")
        
        try:
            # 清除GlobalHydra事故
            try:
                GlobalHydra.instance().clear()
            except:
                pass
            
            # Hydra初始化加载config
            config_dir = os.path.join(base_dir, "1_SAM2_Tiny/code/sam2/configs/sam2")
            if not os.path.exists(config_dir):
                # 作为备选
                config_dir = "/content/drive/MyDrive/vegetation_models_v2/1_SAM2_Tiny/code/sam2/configs/sam2"
            
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                cfg = compose(config_name="sam2_hiera_t.yaml")
            
            model = SAM2Seg(cfg, pretrain_ckpt, device).to(device)
            
            # 尝试加载微调后的权重
            if os.path.exists(finetuned_ckpt):
                ckpt = torch.load(finetuned_ckpt, map_location=device, weights_only=False)
                if 'model_state' in ckpt:
                    model.load_state_dict(ckpt['model_state'])
                else:
                    model.load_state_dict(ckpt)
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
                sam = sam_model_registry['vit_t'](checkpoint=ckpt_path)
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
                    nn.Upsample(size=(1024, 1024), mode='bilinear', align_corners=False)
                )
            
            def forward(self, x):
                return self.seg_head(self.encoder(x))
        
        # 尝试多个权重路径
        ckpt_paths = [
            os.path.join(base_dir, "2_MobileSAM/checkpoints/mobilesam_best.pth"),  # 训练后的权重
            os.path.join(base_dir, "2_MobileSAM/weights/mobile_sam.pt"),  # 预训练权重
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
    try:
        import torch.nn as nn
        from sam2.build_sam import build_sam2_hf
        from hydra.core.global_hydra import GlobalHydra
        
        class SAM21Seg(nn.Module):
            def __init__(self, device, num_classes=2):
                super().__init__()
                # 直接用HF接口加载，自动处理config路径问题
                sam21 = build_sam2_hf(
                    'facebook/sam2.1-hiera-tiny',
                    device=device,
                    mode='train'
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
                    nn.Upsample(size=(1024, 1024), mode='bilinear', align_corners=False)
                )
            
            def forward(self, x):
                features = self.encoder(x)
                # SAM2.1 encoder返回dict，取backbone_fpn最后一层
                if isinstance(features, dict):
                    feat = features['backbone_fpn'][-1]
                else:
                    feat = features
                return self.seg_head(feat)
        
        try:
            # 清除GlobalHydra避免冲突
            try:
                GlobalHydra.instance().clear()
            except:
                pass
            
            model = SAM21Seg(device).to(device)
            
            # 尝试加载微调后的权重
            finetuned_ckpt = os.path.join(base_dir, "5_SAM21_Tiny/checkpoints/sam21tiny_best.pth")
            if os.path.exists(finetuned_ckpt):
                ckpt = torch.load(finetuned_ckpt, map_location=device, weights_only=False)
                if 'model_state' in ckpt:
                    model.load_state_dict(ckpt['model_state'])
                else:
                    model.load_state_dict(ckpt)
                print(f"✅ SAM2.1_Tiny加载成功（微调权重）")
            else:
                print(f"✅ SAM2.1_Tiny加载成功（HuggingFace预训练权重）")
            
            model.eval()
            return model
        except Exception as model_err:
            print(f"⚠️ SAM2.1_Tiny模型加载失败: {model_err}")
            return None
    except Exception as e:
        print(f"⚠️ SAM2.1_Tiny加载失败: {e}")
        return None


def load_mobilesamv2(base_dir, device):
    try:
        import torch.nn as nn
        
        # 尝试多种导入方式
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
                    sam = sam_model_registry['vit_t_mobile'](checkpoint=ckpt_path)
                except:
                    sam = sam_model_registry['vit_t'](checkpoint=ckpt_path)
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
                    nn.Upsample(size=(1024, 1024), mode='bilinear', align_corners=False)
                )
            
            def forward(self, x):
                return self.seg_head(self.encoder(x))
        
        # 尝试多个权重路径
        ckpt_paths = [
            os.path.join(base_dir, "6_MobileSAMV2/checkpoints/mobilesamv2_best.pth"),  # 训练后的权重
            os.path.join(base_dir, "6_MobileSAMV2/weights/mobile_samv2.pt"),  # 预训练权重
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


# ==============================================================================
# BLOCK 7: 加载所有模型（耗时5-15分钟）
# 将此段代码粘贴到Colab第7个cell，运行
# ==============================================================================

print("⏳ Step 7/10: 加载模型（可能需要5-15分钟）...")

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


# ==============================================================================
# BLOCK 8: 定义处理和推理函数
# 将此段代码粘贴到Colab第8个cell，运行
# ==============================================================================

print("⏳ Step 8/10: 定义处理函数...")

def create_overlay_mask(img_bgr, pred_mask, gt_mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    base_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    tp_mask = (pred_mask == 1) & (gt_mask == 1)
    fp_mask = (pred_mask == 1) & (gt_mask == 0)
    fn_mask = (pred_mask == 0) & (gt_mask == 1)
    
    color_mask = np.zeros_like(base_img)
    color_mask[tp_mask] = [0, 0, 255]      # TP: 蓝色
    color_mask[fp_mask] = [0, 255, 0]      # FP: 绿色
    color_mask[fn_mask] = [255, 0, 255]    # FN: 粉色
    
    alpha = 0.5
    overlay_img = np.where(
        color_mask != 0,
        cv2.addWeighted(base_img, 1 - alpha, color_mask, alpha, 0),
        base_img,
    )
    return overlay_img


def preprocess_image(img_path, target_size, device):
    img = Image.open(img_path).convert('RGB')
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

print("✅ 函数定义完成")
print()


# ==============================================================================
# BLOCK 9: 执行推理（耗时10-30分钟）
# 将此段代码粘贴到Colab第9个cell，运行
# ==============================================================================

print("⏳ Step 9/10: 执行推理（可能需要10-30分钟）...")

results = {"Original": []}
for m in MODELS_INFO:
    results[m["name"]] = []

selected_images = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)[:NUM_CASES]

for idx, img_path in enumerate(selected_images):
    print(f"  [{idx+1}/{NUM_CASES}] {os.path.basename(img_path)}")
    
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
    
    for m_info in MODELS_INFO:
        m_name = m_info["name"]
        input_size = m_info["size"]
        
        model = models_dict.get(m_name)
        pred_mask = infer_model(model, img_path, input_size, DEVICE)
        
        overlay_rgb = create_overlay_mask(img_bgr, pred_mask, gt_mask)
        # 缩放到512x512用于显示
        overlay_rgb_512 = cv2.resize(overlay_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        results[m_name].append(overlay_rgb_512)

print("✅ 推理完成")
print()


# ==============================================================================
# BLOCK 10: 生成并保存高清图像（600 DPI，增大字体）
# 将此段代码粘贴到Colab第10个cell，运行
# ==============================================================================

print("⏳ Step 10/10: 生成高清图像（600 DPI）...")

# ============ 📊 调试：检查每个模型的预测统计（修复版本） ============
print("\n📊 模型预测统计:")
for m_info in MODELS_INFO:
    model_name = m_info["name"]
    if model_name in results:
        overlay_imgs = results[model_name]
        if overlay_imgs:
            total_colored_pixels = 0  # 有任何颜色（TP/FP/FN）的像素
            total_pixels = 0
            
            for overlay_img in overlay_imgs:
                # overlay_img是RGB，形状为(512, 512, 3)
                # 计算有颜色的像素（不是完全黑色则表示有标注）
                # 更精确的方式：计算与灰色背景不同的像素
                has_color = (overlay_img != 0).any(axis=2)  # 任何通道非零
                colored_pixels = has_color.sum()
                total_colored_pixels += colored_pixels
                total_pixels += overlay_img.shape[0] * overlay_img.shape[1]
            
            color_ratio = (total_colored_pixels / total_pixels * 100) if total_pixels > 0 else 0
            print(f"  {model_name}: {total_colored_pixels:,} 有标注像素 / {total_pixels:,} 总像素 ({color_ratio:.2f}%)")
        else:
            print(f"  {model_name}: 空结果")
    else:
        print(f"  {model_name}: 未记录")
print()
# ==================================================================

num_cols = len(selected_images)
rows_keys = ["Original"] + [m["name"] for m in MODELS_INFO]
num_rows = len(rows_keys)

cell_w, cell_h = 512, 512
margin_top, margin_left = 120, 80      # 合理边距
h_gap, v_gap = 10, 10

canvas_w = margin_left + num_cols * cell_w + (num_cols - 1) * h_gap + 30
canvas_h = margin_top + num_rows * cell_h + (num_rows - 1) * v_gap + 80

canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

# 字体加载 - 改进版，带更好的容错处理
def try_load_font(fontsize):
    """尝试从多个位置加载字体"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/opentype/dejavu/DejaVuSans-Bold.otf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, fontsize)
        except:
            continue
    
    # 如果都失败了，尝试不带路径的
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    except:
        pass
    
    # 最后使用默认字体
    print(f"⚠️ 警告: 无法加载字体，使用默认字体({fontsize}pt)")
    return ImageFont.load_default()

font_label = try_load_font(60)   # 左侧标签(a)-(g)
font_col = try_load_font(48)     # 上面列号#1-#13
font_title = try_load_font(56)   # 标题
font_legend = try_load_font(40)  # 图例

letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]

# 绘制标题
title = "Case Example - 6 Models Comparison"
title_bbox = draw.textbbox((0, 0), title, font=font_title)
draw.text(
    ((canvas_w - (title_bbox[2] - title_bbox[0])) // 2, 10),
    title,
    font=font_title,
    fill=(0, 0, 0)
)

# 绘制列标题（Case编号）
for c in range(num_cols):
    text = f"#{c+1}"
    bbox = draw.textbbox((0, 0), text, font=font_col)
    cx = margin_left + c * (cell_w + h_gap) + (cell_w - (bbox[2] - bbox[0])) // 2
    cy = margin_top - 60
    draw.text((cx, cy), text, font=font_col, fill=(80, 80, 80))

# 绘制图像网格
for r, r_name in enumerate(rows_keys):
    # 左侧标签
    lx = 8
    ly = margin_top + r * (cell_h + v_gap) + cell_h // 2 - 22
    draw.text((lx, ly), f"({letters[r]})", font=font_label, fill=(0, 0, 0))
    
    # 填充图像
    for c in range(num_cols):
        if r_name in results and c < len(results[r_name]):
            img_np = results[r_name][c]
            img_pil = Image.fromarray(img_np).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            px = margin_left + c * (cell_w + h_gap)
            py = margin_top + r * (cell_h + v_gap)
            canvas.paste(img_pil, (px, py))

# 添加图例（在图下方）
legend_y = margin_top + num_rows * (cell_h + v_gap) + 30
legend_text = "Blue: TP (Correct)  |  Green: FP (False Positive)  |  Pink: FN (False Negative)"
draw.text((margin_left, legend_y), legend_text, font=font_legend, fill=(0, 0, 0))

# 保存高清图像（600 DPI）
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, "fig_6models_13cases_600dpi.png")

try:
    canvas.save(output_file, dpi=(600, 600), optimize=False)
    print("\n" + "="*80)
    print("✅ 完成！")
    print("="*80)
    print(f"📁 输出文件: {output_file}")
    print(f"📊 分辨率: {canvas_w}x{canvas_h}px @ 600DPI")
    print(f"💾 文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print("✨ 颜色含义: 蓝色=TP(正确), 绿色=FP(误检), 粉色=FN(漏检)")
    print("="*80)
except Exception as save_err:
    print(f"❌ 保存图像时出错: {save_err}")
    print("尝试以标准PNG格式保存...")
    try:
        canvas.save(output_file)
        print(f"✅ 已以标准格式保存到: {output_file}")
        print(f"📊 分辨率: {canvas_w}x{canvas_h}px")
        print(f"💾 文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
