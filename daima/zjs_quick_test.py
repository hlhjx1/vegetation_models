# ==============================================================================
# 快速测试脚本 - Colab Cell版本
# 在Colab中试运行这个看看是否有改进
# ==============================================================================

%cd /content/drive/MyDrive/vegetation_models_v2
import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import glob

# 补丁
torch.load = lambda f, *args, **kwargs: (
    dict(kwargs, weights_only=False) if 'weights_only' not in kwargs else kwargs,
    torch.load.__wrapped__(f, *args, **kwargs)
)[1]

DRIVE = "/content/drive/MyDrive/vegetation_models_v2"
ZJS_DATASET_DIR = "/content/drive/MyDrive/datasets/2024-seg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 快速测试 - 设备: {DEVICE}")

# 改变工作目录
_sam2_code_dir = os.path.join(DRIVE, "1_SAM2_Tiny/code")
if os.path.exists(_sam2_code_dir):
    os.chdir(_sam2_code_dir)
    if _sam2_code_dir not in sys.path:
        sys.path.insert(0, _sam2_code_dir)

# ============================================================
# 加载模型（只加载UNet作为示例）
# ============================================================

print("\n📝 第一步: 加载模型")

import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=2,
    activation=None,
).to(DEVICE)

ckpt_path = os.path.join(DRIVE, "7_UNet/checkpoints/unet_best.pth")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    print(f"✅ UNet加载成功")
else:
    print(f"❌ 权重文件不存在: {ckpt_path}")

model.eval()

# ============================================================
# 加载图像
# ============================================================

print("\n📝 第二步: 选择测试图像")

img_dir = os.path.join(ZJS_DATASET_DIR, "JPEGImages")
all_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))

if not all_images:
    print(f"❌ 图像目录: {img_dir} 未找到图像")
else:
    test_img_path = all_images[0]
    print(f"✅ 选择了: {os.path.basename(test_img_path)}")

# ============================================================
# 预处理和推理
# ============================================================

print("\n📝 第三步: 推理")

def preprocess_image(img_path, target_size=512):
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

img_tensor = preprocess_image(test_img_path)

with torch.no_grad():
    output = model(img_tensor)

print(f"  输出类型: {type(output)}")
print(f"  输出形状: {output.shape}")
print(f"  输出范围: [{output.min():.3f}, {output.max():.3f}]")

# ============================================================
# 尝试多种处理方法 - 调试
# ============================================================

print("\n📊 尝试多种处理方法:")

if len(output.shape) == 4 and output.shape[1] == 2:
    
    # 方法1: 原始argmax
    print("\n  🔹 方法1 - argmax:")
    pred1 = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    veg1 = (pred1 > 0).sum()
    print(f"     植被像素: {veg1} ({100*veg1/pred1.size:.2f}%)")
    
    # 方法2: softmax + 0.5
    print("\n  🔹 方法2 - softmax > 0.5:")
    probs = torch.softmax(output, dim=1)
    print(f"     概率范围: [{probs[0,1].min():.3f}, {probs[0,1].max():.3f}]")
    pred2 = (probs[:, 1] > 0.5).squeeze(0).cpu().numpy()
    veg2 = (pred2 > 0).sum()
    print(f"     植被像素: {veg2} ({100*veg2/pred2.size:.2f}%)")
    
    # 方法3: 多个阈值
    print("\n  🔹 方法3 - 显示不同阈值效果:")
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        pred = (probs[:, 1] > threshold).squeeze(0).cpu().numpy()
        veg = (pred > 0).sum()
        print(f"     阈值>{threshold}: {veg} 像素 ({100*veg/pred.size:.2f}%)")

# ============================================================
# 保存结果对比
# ============================================================

print("\n📝 第四步: 保存结果对比")

output_dir = "/content/output_unet_test"
os.makedirs(output_dir, exist_ok=True)

# 原始图像
original_pil = Image.open(test_img_path).convert("RGB")
original_pil = original_pil.resize((512, 512))
original_pil.save(os.path.join(output_dir, "00_original.png"))
print(f"✅ 保存原图")

# 方法1 - argmax
pred1_uint8 = (pred1 * 255).astype(np.uint8)
pred1_img = Image.fromarray(pred1_uint8)
pred1_img.save(os.path.join(output_dir, "01_argmax.png"))
print(f"✅ 保存方法1 (argmax)")

# 方法2 - softmax
pred2_uint8 = (pred2 * 255).astype(np.uint8)
pred2_img = Image.fromarray(pred2_uint8)
pred2_img.save(os.path.join(output_dir, "02_softmax_0.5.png"))
print(f"✅ 保存方法2 (softmax>0.5)")

# 方法3 - 最宽松的阈值
probs_channel = probs[:, 1].squeeze(0).cpu().numpy()
pred3 = (probs_channel > 0.1).cpu().numpy() if isinstance(probs_channel, torch.Tensor) else (probs_channel > 0.1)
pred3_uint8 = (pred3 * 255).astype(np.uint8)
pred3_img = Image.fromarray(pred3_uint8)
pred3_img.save(os.path.join(output_dir, "03_softmax_0.1.png"))
print(f"✅ 保存方法3 (softmax>0.1)")

print(f"\n✨ 测试完成！结果保存在: {output_dir}")
print(f"\n💡 查看这三张预测图:")
print(f"   - 01_argmax.png: 如果这个不是全黑，说明argmax可用")
print(f"   - 02_softmax_0.5.png: 如果这个是全黑，可能阈值太高")
print(f"   - 03_softmax_0.1.png: 较低阈值看是否有植被检测")
print(f"\n🔧 根据结果，调整zjs_single_inference_final.py中的阈值")
