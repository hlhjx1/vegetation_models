# ==============================================================================
# Part 2: 定义所有模型加载函数（第2个 Colab Cell）
# ==============================================================================

print("⏳ Part 2/3: 定义模型加载函数...")

# ============ 关键Hydra初始化：清除之前的状态 + 改变工作目录 ============
from hydra.core.global_hydra import GlobalHydra

try:
    GlobalHydra.instance().clear()
    print("✅ GlobalHydra 已清除")
except:
    pass

_original_cwd = os.getcwd()
_sam2_code_dir = os.path.join(DRIVE, "1_SAM2_Tiny/code")
if os.path.exists(_sam2_code_dir):
    os.chdir(_sam2_code_dir)
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
        from hydra import initialize_config_dir, compose

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

        pretrain_ckpt = os.path.join(base_dir, "1_SAM2_Tiny/weights/sam2_hiera_tiny.pt")

        try:
            # 尝试用 initialize_config_dir 初始化 Hydra
            config_dir = os.path.join(base_dir, "1_SAM2_Tiny/code/sam2/configs/sam2")
            if os.path.exists(config_dir):
                with initialize_config_dir(version_base=None, config_dir=config_dir):
                    model = SAM2Seg(
                        "sam2_hiera_t.yaml", pretrain_ckpt, num_classes=2
                    ).to(device)
            else:
                # 备选：使用工作目录方式
                model = SAM2Seg(
                    "sam2/sam2_hiera_t.yaml", pretrain_ckpt, num_classes=2
                ).to(device)
        except Exception as hydra_err:
            print(f"  Hydra初始化失败，尝试备选方案: {hydra_err}")
            model = SAM2Seg("sam2/sam2_hiera_t.yaml", pretrain_ckpt, num_classes=2).to(
                device
            )

        best_ckpt = os.path.join(base_dir, "1_SAM2_Tiny/checkpoints/sam2tiny_best.pth")
        if os.path.exists(best_ckpt):
            try:
                ckpt_state = torch.load(
                    best_ckpt, map_location=device, weights_only=False
                )
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
    """SAM2.1-Tiny"""
    try:
        import torch.nn as nn
        from sam2.build_sam import build_sam2_hf
        from hydra import initialize_config_dir, compose

        class SAM21Seg(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                try:
                    # 优先尝试本地加载
                    config_dir = os.path.join(
                        base_dir, "1_SAM2_Tiny/code/sam2.1/configs/sam2.1"
                    )
                    with initialize_config_dir(
                        version_base=None, config_dir=config_dir
                    ):
                        from sam2.build_sam import build_sam2

                        sam21 = build_sam2(
                            "sam2.1_hiera_t.yaml",
                            os.path.join(
                                base_dir, "5_SAM21_Tiny/weights/sam2.1_hiera_tiny.pt"
                            ),
                            device=device,
                            mode="train",
                        )
                except:
                    # 备选：从 HuggingFace 加载
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
                ckpt_state = torch.load(
                    best_ckpt, map_location=device, weights_only=False
                )
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


print("✅ 所有函数定义完成！")
print(f"✅ Part 2 完成！下一步运行 zjs_推理_Part3_加载推理.py")
