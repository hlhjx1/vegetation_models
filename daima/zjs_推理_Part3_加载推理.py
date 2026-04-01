# ==============================================================================
# Part 3: 加载模型、推理、保存结果（第3个 Colab Cell）
# ==============================================================================

print("=" * 80)
print("🎯 紫金山单张图像推理 - 6模型预测")
print("=" * 80)

print(f"\n⏳ 加载所有6个模型...")

# 尝试重新清除 GlobalHydra 事故
try:
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
except:
    pass

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
# 彩色叠加函数（基于GT标签的错误分析）
# ============================================================


def create_overlay_mask(img_bgr, pred_mask, gt_mask):
    """
    生成彩色对比图：
    - 蓝色 (TP)   : 预测正确
    - 绿色 (FP)   : 误检测
    - 粉色 (FN)   : 漏检
    """
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
    return overlay_img.astype(np.uint8)


# ============================================================
# 推理函数
# ============================================================


def preprocess_image(img_path, target_size, device):
    """图像预处理 - ImageNet标准化"""
    img = Image.open(img_path).convert("RGB")
    tf = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tf(img).unsqueeze(0).to(device)


def infer_model(model, img_path, img_size, device):
    """使用模型进行推理"""
    if model is None:
        return np.zeros((512, 512), dtype=np.uint8)

    try:
        img_tensor = preprocess_image(img_path, img_size, device)
        with torch.no_grad():
            output = model(img_tensor)
            preds = (
                torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            )

        if preds.shape != (512, 512):
            preds = cv2.resize(
                preds.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST
            )

        return preds

    except Exception as e:
        print(f"    ⚠️ 推理异常: {e}")
        return np.zeros((512, 512), dtype=np.uint8)


# ============================================================
# Step 1: 选择图像
# ============================================================

print(f"⏳ Step 1/3: 选择图像...")

img_dir = os.path.join(ZJS_DATASET_DIR, "JPEGImages")
mask_dir = os.path.join(ZJS_DATASET_DIR, "SegmentationClass")

if not os.path.exists(img_dir):
    print(f"  ❌ 图像目录不存在: {img_dir}")
else:
    all_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    if not all_images:
        print(f"  ❌ 未找到PNG图像")
    else:
        selected_image = all_images[0]
        image_name = os.path.basename(selected_image).replace(".png", "")

        print(f"  ✅ 找到 {len(all_images)} 张图像")
        print(f"  ✅ 选中第1张: {image_name}.png")

        img_bgr = cv2.imread(selected_image)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 加载对应的Ground Truth标签
        mask_path = os.path.join(mask_dir, f"{image_name}.png")
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            print(f"  ✅ Ground Truth标签已加载")
        else:
            gt_mask = None
            print(f"  ⚠️ Ground Truth标签不存在: {mask_path}")

        output_original = os.path.join(OUTPUT_DIR, "01_original_image.png")
        Image.fromarray(img_rgb).save(output_original)
        print(f"\n  💾 原图已保存:")
        print(f"     {output_original}")
        print(f"     分辨率: {img_rgb.shape[1]}×{img_rgb.shape[0]}")

        # ============================================================
        # Step 2: 推理
        # ============================================================

        print(f"\n⏳ Step 2/3: 模型推理...")

        predictions = {}
        pred_stats = {}

        for m_info in MODELS_INFO:
            m_name = m_info["name"]
            input_size = m_info["size"]

            model = models_dict.get(m_name)

            print(f"  推理 {m_name:15s}...", end=" ")
            pred_mask = infer_model(model, selected_image, input_size, DEVICE)

            veg_pixels = (pred_mask > 0).sum()
            total_pixels = pred_mask.size
            coverage = (veg_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            pred_stats[m_name] = coverage

            # 保存预测mask用于后续处理
            predictions[m_name] = pred_mask

            print(f"✅ (植被覆盖率: {coverage:.2f}%)")

        # ============================================================
        # Step 3: 生成彩色对比图并保存
        # ============================================================

        print(f"\n⏳ Step 3/3: 生成彩色对比图并保存...")

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
                # 生成彩色对比图（如果有GT标签）
                if gt_mask is not None:
                    pred_mask = predictions[m_name]
                    overlay_img = create_overlay_mask(img_bgr, pred_mask, gt_mask)
                    output_file = os.path.join(
                        OUTPUT_DIR, f"{save_prefix}_prediction.png"
                    )
                    cv2.imwrite(
                        output_file, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
                    )
                else:
                    # 如果没有GT标签，直接保存二值图
                    pred_mask = predictions[m_name]
                    pred_rgb = np.stack(
                        [pred_mask * 255, pred_mask * 255, pred_mask * 255], axis=-1
                    ).astype(np.uint8)
                    output_file = os.path.join(
                        OUTPUT_DIR, f"{save_prefix}_prediction.png"
                    )
                    Image.fromarray(pred_rgb).save(output_file)

                coverage = pred_stats.get(m_name, 0)
                print(
                    f"  💾 {m_name:15s} → {save_prefix}_prediction.png (覆盖率: {coverage:.2f}%)"
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
        print(f"   ✓ 02_unet_prediction.png         （UNet预测 - 彩色对比图）")
        print(f"   ✓ 03_deeplabv3plus_prediction.png（DeepLabV3+预测 - 彩色对比图）")
        print(f"   ✓ 04_sam2_tiny_prediction.png    （SAM2_Tiny预测 - 彩色对比图）")
        print(f"   ✓ 05_mobilesam_prediction.png    （MobileSAM预测 - 彩色对比图）")
        print(f"   ✓ 06_sam21_tiny_prediction.png   （SAM2.1_Tiny预测 - 彩色对比图）")
        print(f"   ✓ 07_mobilesamv2_prediction.png  （MobileSAMV2预测 - 彩色对比图）")

        print(f"\n📊 植被覆盖率统计：")
        for m_name, coverage in pred_stats.items():
            print(f"   {m_name:15s}: {coverage:7.2f}%")

        if gt_mask is not None:
            print(f"\n🎨 彩色标记表：")
            print(f"   • 蓝色  = TP (True Positive)  - 正确检测植被")
            print(f"   • 绿色  = FP (False Positive) - 误检测（背景被检为植被）")
            print(f"   • 粉色  = FN (False Negative) - 漏检（植被被检为背景）")
        else:
            print(f"\n💡 预测值映射:")
            print(f"   • 黑色 (0) = 背景区域")
            print(f"   • 白色 (255) = 植被区域")

        print("=" * 80)
