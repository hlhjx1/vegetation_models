# ==============================================================================
# BLOCK 6: 模型加载函数 - 最终版（解决所有环境问题）
# ==============================================================================

print("⏳ Step 6/10: 定义模型加载函数...")

# ============ 关键1: 改变工作目录使Hydra能找到config ============
_original_cwd = os.getcwd()
_sam2_code_dir = os.path.join(BASE_DIR, "1_SAM2_Tiny/code")
if os.path.exists(_sam2_code_dir):
    os.chdir(_sam2_code_dir)
    import sys

    if _sam2_code_dir not in sys.path:
        sys.path.insert(0, _sam2_code_dir)
    print(f"✅ 工作目录已改为: {_sam2_code_dir}")
# ====================================================


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
        import traceback

        traceback.print_exc()
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
        import traceback

        traceback.print_exc()
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


print("✅ 函数定义完成")
print()
