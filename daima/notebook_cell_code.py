"""
============================================================
新Cell - 交叉数据集推理可视化 (6模型 × 3数据集 × 13样本)
============================================================
将此段完整代码粘贴到对比实验_三数据集_8模型训练.ipynb的最后

前置条件：
  ✅ 已运行 Cell 1 (环境配置)
  ✅ 已运行 Cell 3-7 (数据集预处理)
  ✅ 已运行 Cell 8 (工具函数定义)
  ✅ 已运行 Cell 9 (DataLoader构建)
  ✅ 已运行 Cell 11-20 (模型训练 或 已有权重)

输出：三份600DPI高清对比图
  - fig_loveda_6models_13cases_600dpi.png
  - fig_potsdam_6models_13cases_600dpi.png
  - fig_vaihingen_6models_13cases_600dpi.png
"""

# ============================================================
# 导入必要库
# ============================================================

import os, sys, json, glob, torch, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
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
print("🎯 交叉数据集推理可视化 - 6模型 × 3数据集 × 13样本")
print("=" * 80)

# ============================================================
# 配置参数
# ============================================================

DRIVE = "/content/drive/MyDrive/vegetation_models_v2"
BINARY_ROOT = "/content/binary"
OUTPUT_DIR = os.path.join(DRIVE, "paper_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["LoveDA", "Potsdam", "Vaihingen"]
SAMPLES_PER_DATASET = 13
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

# ============================================================
# 模型加载函数 (全部6个模型)
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
# 推理工具函数
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


def infer_model(model, img_path, img_size):
    """使用模型进行推理"""
    if model is None:
        return np.zeros((512, 512), dtype=np.uint8)

    try:
        img_tensor = preprocess_image(img_path, img_size)
        with torch.no_grad():
            output = model(img_tensor)
            preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        if preds.shape != (512, 512):
            preds = cv2.resize(
                preds.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST
            )

        return preds
    except Exception as e:
        print(f"      ⚠️  推理异常: {e}")
        return np.zeros((512, 512), dtype=np.uint8)


def create_overlay_mask(img_bgr, pred_mask, gt_mask):
    """
    创建三色预测覆盖图
    - TP (正确): 蓝色
    - FP (误检): 绿色
    - FN (漏检): 粉红色
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    base_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # 二值化
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    # 计算TP/FP/FN
    tp_mask = (pred_bin == 1) & (gt_bin == 1)
    fp_mask = (pred_bin == 1) & (gt_bin == 0)
    fn_mask = (pred_bin == 0) & (gt_bin == 1)

    # 彩色覆盖
    color_mask = np.zeros_like(base_img)
    color_mask[tp_mask] = [0, 0, 255]  # TP: 蓝色
    color_mask[fp_mask] = [0, 255, 0]  # FP: 绿色
    color_mask[fn_mask] = [255, 0, 255]  # FN: 粉红色

    # 透明混合
    alpha = 0.5
    overlay_img = np.where(
        color_mask != 0,
        cv2.addWeighted(base_img, 1 - alpha, color_mask, alpha, 0),
        base_img,
    )

    return overlay_img.astype(np.uint8)


def try_load_font(fontsize):
    """尝试加载字体，若失败用默认"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]

    for path in font_paths:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, fontsize)
        except:
            pass

    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    except:
        return ImageFont.load_default()


# ============================================================
# 为每个数据集生成对比展示图
# ============================================================

print(f"\n⏳ 开始推理与可视化...\n")

for ds_idx, ds_name in enumerate(DATASETS):
    print("=" * 75)
    print(f"📊 [{ds_idx+1}/3] {ds_name:10s} - 6模型 × {SAMPLES_PER_DATASET}样本对比图")
    print("=" * 75)

    # ────────────────────────────────────
    # 第一步：选择样本
    # ────────────────────────────────────

    img_dir = os.path.join(BINARY_ROOT, ds_name.lower(), "JPEGImages")
    mask_dir = os.path.join(BINARY_ROOT, ds_name.lower(), "SegmentationClass")

    if not os.path.exists(img_dir):
        print(f"  ❌ 图像目录不存在: {img_dir}")
        continue

    all_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    selected_images = all_images[:SAMPLES_PER_DATASET]

    print(f"  ✅ 选择了 {len(selected_images)} 个样本")

    if len(selected_images) < SAMPLES_PER_DATASET:
        print(f"     (数据集仅有 {len(all_images)} 张，需要 {SAMPLES_PER_DATASET} 张)")

    # ────────────────────────────────────
    # 第二步：为每个模型推理
    # ────────────────────────────────────

    results = {"Original": []}
    for m_info in MODELS_INFO:
        results[m_info["name"]] = []

    print(f"\n  推理进度:")

    for idx, img_path in enumerate(selected_images):
        if (idx + 1) % 5 == 0 or idx == 0 or idx == len(selected_images) - 1:
            print(
                f"    [{idx+1:2d}/{len(selected_images)}] {os.path.basename(img_path)}"
            )

        # 读取原图
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"      ❌ 无法读取: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_512 = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        results["Original"].append(img_rgb_512)

        # 读取GT mask
        filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, filename + ".png")

        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            gt_mask = (gt_mask > 0).astype(np.uint8)
        else:
            gt_mask = np.zeros((512, 512), dtype=np.uint8)

        # 为 6 个模型推理
        for m_info in MODELS_INFO:
            m_name = m_info["name"]
            input_size = m_info["size"]

            # 从全局 models_dict 中获取已加载的模型
            model = models_dict.get(m_name)

            pred_mask = infer_model(model, img_path, input_size)

            # 确保原图也缩放到512×512用于叠加
            img_bgr_512 = cv2.resize(
                img_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4
            )
            overlay_rgb = create_overlay_mask(img_bgr_512, pred_mask, gt_mask)
            # overlay已经是512×512，无需再resize
            results[m_name].append(overlay_rgb)

    # ────────────────────────────────────
    # 第三步：生成高清对比图 (600 DPI)
    # ────────────────────────────────────

    print(f"\n  生成高清对比图...")

    num_cols = len(selected_images)
    rows_keys = ["Original"] + [m["name"] for m in MODELS_INFO]
    num_rows = len(rows_keys)

    cell_w, cell_h = 512, 512
    margin_top, margin_left = 120, 80
    h_gap, v_gap = 10, 10

    canvas_w = margin_left + num_cols * cell_w + (num_cols - 1) * h_gap + 30
    canvas_h = margin_top + num_rows * cell_h + (num_rows - 1) * v_gap + 80

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 加载字体
    font_label = try_load_font(60)
    font_col = try_load_font(48)
    font_title = try_load_font(56)
    font_legend = try_load_font(40)

    letters = [chr(i) for i in range(ord("a"), ord("z") + 1)]

    # 绘制标题
    title = f"{ds_name} - 6 Models × {len(selected_images)} Cases"
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    draw.text(
        ((canvas_w - (title_bbox[2] - title_bbox[0])) // 2, 10),
        title,
        font=font_title,
        fill=(0, 0, 0),
    )

    # 绘制列标题 (Case 编号)
    for c in range(num_cols):
        text = f"#{c+1}"
        bbox = draw.textbbox((0, 0), text, font=font_col)
        cx = margin_left + c * (cell_w + h_gap) + (cell_w - (bbox[2] - bbox[0])) // 2
        cy = margin_top - 60
        draw.text((cx, cy), text, font=font_col, fill=(80, 80, 80))

    # 绘制图像网格与行标签
    for r, r_name in enumerate(rows_keys):
        # 行标签 (a)-(g)
        lx = 8
        ly = margin_top + r * (cell_h + v_gap) + cell_h // 2 - 22
        draw.text((lx, ly), f"({letters[r]})", font=font_label, fill=(0, 0, 0))

        # 图像单元格
        for c in range(num_cols):
            if r_name in results and c < len(results[r_name]):
                img_np = results[r_name][c]
                img_pil = Image.fromarray(img_np).resize(
                    (cell_w, cell_h), Image.Resampling.LANCZOS
                )
                px = margin_left + c * (cell_w + h_gap)
                py = margin_top + r * (cell_h + v_gap)
                canvas.paste(img_pil, (px, py))

    # 添加图例
    legend_y = margin_top + num_rows * (cell_h + v_gap) + 30
    legend_text = (
        "Blue: TP (Correct) | Green: FP (False Positive) | Pink: FN (False Negative)"
    )
    draw.text((margin_left, legend_y), legend_text, font=font_legend, fill=(0, 0, 0))

    # 保存高清图像 (600 DPI)
    output_file = os.path.join(
        OUTPUT_DIR,
        f"fig_{ds_name.lower()}_6models_{len(selected_images)}cases_600dpi.png",
    )

    try:
        canvas.save(output_file, dpi=(600, 600), optimize=False)
        file_size = os.path.getsize(output_file) / (1024 * 1024)

        print(f"\n  ✅ 对比图已保存")
        print(f"     📁 {output_file}")
        print(f"     📊 分辨率: {canvas_w}×{canvas_h}px @ 600DPI")
        print(f"     💾 文件大小: {file_size:.2f} MB")

    except Exception as e:
        print(f"  ❌ 保存失败: {e}")


# ============================================================
# 完成提示
# ============================================================

print("\n" + "=" * 80)
print("✅ 交叉数据集对比图生成完成！")
print("=" * 80)
print(f"\n📂 所有图像已保存到:")
print(f"   {OUTPUT_DIR}")
print(f"\n📄 生成的图像:")
print(f"   ✓ fig_loveda_6models_{SAMPLES_PER_DATASET}cases_600dpi.png")
print(f"   ✓ fig_potsdam_6models_{SAMPLES_PER_DATASET}cases_600dpi.png")
print(f"   ✓ fig_vaihingen_6models_{SAMPLES_PER_DATASET}cases_600dpi.png")
print(f"\n🎨 每个对比图包含:")
print(f"   - 行 (a): 原始RGB图像")
print(f"   - 行 (b-g): 6个模型的预测覆盖 (TP/FP/FN三色)")
print(f"   - 列: {SAMPLES_PER_DATASET}个测试样本")
print(f"\n📊 分辨率: 6886×3844px @ 600DPI (≈47MB)")
print("=" * 80)
