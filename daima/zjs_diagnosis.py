# ==============================================================================
# 诊断脚本: 检查为什么预测全黑
# ==============================================================================
"""
用来诊断模型输出的问题
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

torch.load = lambda f, *args, **kwargs: (
    dict(kwargs, weights_only=False) if "weights_only" not in kwargs else kwargs,
    torch.load.__wrapped__(f, *args, **kwargs),
)[1]

DRIVE = "/content/drive/MyDrive/vegetation_models_v2"
ZJS_DATASET_DIR = "/content/drive/MyDrive/datasets/2024-seg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("🔍 诊断模式: 检查模型推理输出")
print("=" * 80)

# 改变工作目录
_sam2_code_dir = os.path.join(DRIVE, "1_SAM2_Tiny/code")
if os.path.exists(_sam2_code_dir):
    os.chdir(_sam2_code_dir)
    if _sam2_code_dir not in sys.path:
        sys.path.insert(0, _sam2_code_dir)

# 加载一个模型进行诊断
print("\n⏳ 加载UNet进行诊断...")

try:
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

    model.eval()

except Exception as e:
    print(f"❌ UNet加载失败: {e}")
    sys.exit(1)

# 选择第一张测试图
print("\n⏳ 选择测试图像...")

img_dir = os.path.join(ZJS_DATASET_DIR, "JPEGImages")
import glob

all_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))

if not all_images:
    print(f"❌ 未找到图像")
    sys.exit(1)

test_img_path = all_images[0]
print(f"✅ 选中测试图: {os.path.basename(test_img_path)}")

# 预处理
print("\n⏳ 预处理图像...")

img = Image.open(test_img_path).convert("RGB")
tf = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
img_tensor = tf(img).unsqueeze(0).to(DEVICE)

print(f"  输入张量形状: {img_tensor.shape}")
print(f"  输入值范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

# 推理
print("\n⏳ 推理...")

with torch.no_grad():
    output = model(img_tensor)

print(f"\n📊 输出诊断信息:")
print(f"  输出类型: {type(output)}")
print(f"  输出形状: {output.shape}")
print(f"  输出数据类型: {output.dtype}")
print(f"  输出值范围: [{output.min():.3f}, {output.max():.3f}]")
print(f"  输出平均值: {output.mean():.3f}")
print(f"  输出方差: {output.var():.3f}")

# 分析通道
if len(output.shape) == 4 and output.shape[1] == 2:
    chan0 = output[0, 0]
    chan1 = output[0, 1]
    print(f"\n  通道0 (背景):")
    print(f"    范围: [{chan0.min():.3f}, {chan0.max():.3f}]")
    print(f"    平均: {chan0.mean():.3f}")
    print(f"  通道1 (植被):")
    print(f"    范围: [{chan1.min():.3f}, {chan1.max():.3f}]")
    print(f"    平均: {chan1.mean():.3f}")

# 尝试不同的阈值方案
print(f"\n🔧 尝试不同的处理方案:")

# 方案1: argmax
if len(output.shape) == 4 and output.shape[1] == 2:
    pred_argmax = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    veg_pixels_1 = (pred_argmax > 0).sum()
    print(
        f"  方案1 (argmax): 植被像素数 = {veg_pixels_1} / {pred_argmax.size} ({100*veg_pixels_1/pred_argmax.size:.2f}%)"
    )

    # 方案2: softmax + 0.5阈值
    probs = torch.softmax(output, dim=1)
    pred_softmax = (probs[0, 1] > 0.5).cpu().numpy().astype(np.uint8)
    veg_pixels_2 = (pred_softmax > 0).sum()
    print(
        f"  方案2 (softmax>0.5): 植被像素数 = {veg_pixels_2} / {pred_softmax.size} ({100*veg_pixels_2/pred_softmax.size:.2f}%)"
    )

    # 方案3: 原始值 > 某个值
    raw_vegetation = output[0, 1].squeeze().cpu().numpy()
    for threshold in [0, 0.1, 0.5, 1.0]:
        veg_pixels_3 = (raw_vegetation > threshold).sum()
        print(
            f"  方案3 (raw>>{threshold}): 植被像素数 = {veg_pixels_3} / {raw_vegetation.size} ({100*veg_pixels_3/raw_vegetation.size:.2f}%)"
        )

print("\n" + "=" * 80)
print("💡 诊断完成")
print("=" * 80)
print(f"\n✅ 如果所有方案都显示植被像素数为0，说明模型推理输出有问题")
print(f"   可能原因:")
print(f"   1. 模型权重加载不正确")
print(f"   2. 模型根本没有学到植被检测")
print(f"   3. 输入预处理有问题")
print(f"\n⚠️ 如果某个方案显示了非零的植被像素，用那种方案处理即可")
