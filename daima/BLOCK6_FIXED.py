# ==============================================================================
# BLOCK 6: 模型加载函数（直接套用training notebook，修复Hydra冲突）
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
            activation=None,
        ).to(device)
        ckpt_path = os.path.join(base_dir, "7_UNet/checkpoints/unet_best.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(
                ckpt["model_state"] if "model_state" in ckpt else ckpt
            )
            print(f"✅ UNet权重加载成功")
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
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(
                ckpt["model_state"] if "model_state" in ckpt else ckpt
            )
            print(f"✅ DeepLabV3+权重加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ DeepLabV3+加载失败: {e}")
        return None


def load_sam2_tiny(base_dir, device):
    try:
        import torch.nn as nn
        from sam2.build_sam import build_sam2
        from hydra import compose
        from hydra.core.global_hydra import GlobalHydra

        # 清理GlobalHydra避免冲突
        GlobalHydra.instance().clear()

        class SAM2Seg(nn.Module):
            def __init__(self, cfg_name, ckpt_path, num_classes=2):
                super().__init__()
                # 直接用cfg_name调用build_sam2，它内部会处理Hydra
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
                self.encoder.eval()
                for p in self.encoder.parameters():
                    p.requires_grad = False

            def forward(self, x):
                with torch.no_grad():
                    features = self.encoder(x)
                # SAM2 encoder返回dict，取backbone_fpn最后一层
                if isinstance(features, dict):
                    feat = features["backbone_fpn"][-1]
                else:
                    feat = features
                return self.seg_head(feat)

        ckpt = os.path.join(base_dir, "1_SAM2_Tiny/weights/sam2_hiera_tiny.pt")
        if not os.path.exists(ckpt):
            print(f"⚠️ SAM2_Tiny权重找不到")
            return None

        model = SAM2Seg("sam2_hiera_t.yaml", ckpt, num_classes=2).to(device)

        # 加载checkpoint中的seg_head权重
        best_ckpt = os.path.join(base_dir, "1_SAM2_Tiny/checkpoints/sam2tiny_best.pth")
        if os.path.exists(best_ckpt):
            try:
                ckpt_state = torch.load(
                    best_ckpt, map_location=device, weights_only=False
                )
                model.load_state_dict(
                    ckpt_state["model_state"]
                    if "model_state" in ckpt_state
                    else ckpt_state
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
    try:
        import torch.nn as nn
        from mobile_sam import sam_model_registry

        class MobileSAMSeg(nn.Module):
            def __init__(self, ckpt_path, num_classes=2):
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
                self.encoder.eval()
                for p in self.encoder.parameters():
                    p.requires_grad = False

            def forward(self, x):
                with torch.no_grad():
                    feat = self.encoder(x)
                return self.seg_head(feat)

        ckpt_paths = [
            os.path.join(base_dir, "2_MobileSAM/checkpoints/mobilesam_best.pth"),
        ]
        ckpt = None
        for p in ckpt_paths:
            if os.path.exists(p):
                ckpt = p
                break

        if ckpt is None:
            print(f"⚠️ MobileSAM权重找不到")
            return None

        model = MobileSAMSeg(ckpt, num_classes=2).to(device)
        print(f"✅ MobileSAM加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ MobileSAM加载失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_sam21_tiny(base_dir, device):
    try:
        import torch.nn as nn
        from sam2.build_sam import build_sam2_hf
        from hydra.core.global_hydra import GlobalHydra

        # 清理GlobalHydra避免冲突
        GlobalHydra.instance().clear()

        class SAM21Seg(nn.Module):
            def __init__(self, num_classes=2):
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
                self.encoder.eval()
                for p in self.encoder.parameters():
                    p.requires_grad = False

            def forward(self, x):
                with torch.no_grad():
                    features = self.encoder(x)
                # SAM2 encoder返回dict或tensor
                if isinstance(features, dict):
                    feat = features["backbone_fpn"][-1]
                else:
                    feat = features
                return self.seg_head(feat)

        model = SAM21Seg(num_classes=2).to(device)

        # 加载checkpoint中的seg_head权重
        best_ckpt = os.path.join(
            base_dir, "5_SAM21_Tiny/checkpoints/sam21tiny_best.pth"
        )
        if os.path.exists(best_ckpt):
            try:
                ckpt_state = torch.load(
                    best_ckpt, map_location=device, weights_only=False
                )
                model.load_state_dict(
                    ckpt_state["model_state"]
                    if "model_state" in ckpt_state
                    else ckpt_state
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
    try:
        import torch.nn as nn
        from mobile_sam import sam_model_registry

        class MobileSAMV2Seg(nn.Module):
            def __init__(self, ckpt_path, num_classes=2):
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
                self.encoder.eval()
                for p in self.encoder.parameters():
                    p.requires_grad = False

            def forward(self, x):
                with torch.no_grad():
                    feat = self.encoder(x)
                return self.seg_head(feat)

        ckpt_paths = [
            os.path.join(base_dir, "6_MobileSAMV2/checkpoints/mobilesamv2_best.pth"),
        ]
        ckpt = None
        for p in ckpt_paths:
            if os.path.exists(p):
                ckpt = p
                break

        if ckpt is None:
            print(f"⚠️ MobileSAMV2权重找不到")
            return None

        model = MobileSAMV2Seg(ckpt, num_classes=2).to(device)
        print(f"✅ MobileSAMV2加载成功")
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ MobileSAMV2加载失败: {e}")
        import traceback

        traceback.print_exc()
        return None


print("✅ 函数定义完成")
print()
