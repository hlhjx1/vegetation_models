# Colab完整集成指南：紫金山单张推理

## 📋 集成到现有Notebook

### 步骤1: 打开Notebook

打开位置：`daima/对比实验_三数据集_8模型训练.ipynb` 

### 步骤2: 定位最后一个Cell

页面向下滚动至最后，找到最后一个Cell

### 步骤3: 添加新Cell

在最后Cell下方：
- 点击 **"+ 代码"**（Insert > Code cell）
- 新建一个代码单元

### 步骤4: 复制代码（方式A：推荐）

打开本地文件：`zjs_single_inference_colab_cell.py`

**全选所有代码** (Ctrl+A)，**复制** (Ctrl+C)

在Colab新Cell中**粘贴** (Ctrl+V)

### 步骤5: 运行Cell

```
按 Shift + Enter
```

---

## 🔌 集成到Notebook的另一种方式（方式B）

在Colab新Cell中直接输入：

```python
# 从本地Python文件运行
%run /content/drive/MyDrive/vegetation_models_v2/daima/zjs_single_inference_colab_cell.py
```

然后运行 Shift+Enter

---

## ✅ 检查清单：运行前必须验证

| 检查项 | 状态 | 说明 |
|-------|------|------|
| Cell 1-9 已执行 | ☐ | 必须！这样 `models_dict` 才会初始化 |
| Drive已挂载 | ☐ | 检查左上角是否显示完整路径 |
| 紫金山数据已上传 | ☐ | 数据应在 `/content/drive/MyDrive/datasets/2024-seg/` |
| GPU已启用 | ☐ | 可选，但推荐（运行速度快10倍） |

### ✨ 启用GPU步骤

1. 点击 **"运行时"** → **"更改运行时类型"**
2. 选择 GPU (T4 或 A100)
3. 点击 **保存**
4. Cell会自动重启

---

## 📊 运行结果预期

### 正常输出 (7-10秒)

```
============================================================
🎯 紫金山单张图像推理 - 6模型预测
============================================================

📦 配置信息:
  紫金山数据集: /content/drive/MyDrive/datasets/2024-seg
  输出目录: /content/drive/MyDrive/vegetation_models_v2/output_zjs
  设备: cuda

⏳ Step 1/3: 选择图像...
  ✅ 找到 67 张图像
  ✅ 选中第1张: omap_2024_000000.png
  💾 原图已保存: .../output_zjs/01_original_image.png

⏳ Step 2/3: 模型推理...
  推理 UNet              ... ✅
  推理 DeepLabV3+        ... ✅
  推理 SAM2_Tiny         ... ✅
  推理 MobileSAM         ... ✅
  推理 SAM2.1_Tiny       ... ✅
  推理 MobileSAMV2       ... ✅

⏳ Step 3/3: 保存预测结果...
  💾 UNet                → ...02_unet_prediction.png
  💾 DeepLabV3+          → ...03_deeplabv3plus_prediction.png
  💾 SAM2_Tiny           → ...04_sam2_tiny_prediction.png
  💾 MobileSAM           → ...05_mobilesam_prediction.png
  💾 SAM2.1_Tiny         → ...06_sam21_tiny_prediction.png
  💾 MobileSAMV2         → ...07_mobilesamv2_prediction.png

============================================================
✅ 推理完成！
============================================================
```

### 错误排查

#### 错误1: KeyError
```
KeyError: 'LoveDA'
```
**原因**: Cell 1-9尚未运行  
**修复**: 向上滚动，执行Cell 1-9

#### 错误2: FileNotFoundError
```
FileNotFoundError: [Errno 2] No such file or directory: '/content/.../2024-seg'
```
**原因**: 紫金山数据集不在指定位置  
**修复**: 
- 检查文件是否已上传
- 验证路径是否正确

#### 错误3: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**原因**: GPU内存不足  
**修复**: 在代码顶部改为使用CPU
```python
DEVICE = "cpu"  # 改这一行
```

---

## 🎯 下一步（附加分析）

Cell推理完成后，可添加以下代码进行可视化：

### 代码片段1: 显示7张图排成一行

```python
import matplotlib.pyplot as plt
from PIL import Image
import os

output_dir = "/content/drive/MyDrive/vegetation_models_v2/output_zjs"

fig, axes = plt.subplots(1, 7, figsize=(21, 3))
fig.suptitle('Zijinshan 单张图像 - 6模型对比', fontsize=14, y=1.02)

image_files = [
    "01_original_image.png",
    "02_unet_prediction.png",
    "03_deeplabv3plus_prediction.png",
    "04_sam2_tiny_prediction.png",
    "05_mobilesam_prediction.png",
    "06_sam21_tiny_prediction.png",
    "07_mobilesamv2_prediction.png",
]

titles = ["原图", "UNet", "DeepLabV3+", "SAM2_T", "MobileSAM", "SAM2.1_T", "MobileV2"]

for ax, img_file, title in zip(axes, image_files, titles):
    img = Image.open(os.path.join(output_dir, img_file))
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "00_comparison.png"), dpi=150, bbox_inches='tight')
print(f"✅ 对比图已保存: {output_dir}/00_comparison.png")
plt.show()
```

### 代码片段2: 计算植被覆盖率

```python
import cv2
import numpy as np
import os

output_dir = "/content/drive/MyDrive/vegetation_models_v2/output_zjs"

print("\n📊 植被覆盖率统计:")
print("-" * 60)

model_names = ["UNet", "DeepLabV3+", "SAM2_Tiny", "MobileSAM", "SAM2.1_Tiny", "MobileSAMV2"]
pred_files = [
    "02_unet_prediction.png",
    "03_deeplabv3plus_prediction.png",
    "04_sam2_tiny_prediction.png",
    "05_mobilesam_prediction.png",
    "06_sam21_tiny_prediction.png",
    "07_mobilesamv2_prediction.png",
]

for model_name, pred_file in zip(model_names, pred_files):
    img = cv2.imread(os.path.join(output_dir, pred_file), cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        total_pixels = img.size
        vegetation_pixels = np.count_nonzero(img > 128)  # 白色像素
        coverage = (vegetation_pixels / total_pixels) * 100
        
        print(f"{model_name:15s}: {coverage:6.2f}% ({vegetation_pixels:7d}/{total_pixels} 像素)")

print("-" * 60)
```

---

## 📁 输出文件位置

所有生成文件保存在：

```
Google Drive/vegetation_models_v2/output_zjs/
```

### 文件列表

```
output_zjs/
├── 01_original_image.png          [512×512, RGB原图]
├── 02_unet_prediction.png         [512×512, 黑白预测]
├── 03_deeplabv3plus_prediction.png
├── 04_sam2_tiny_prediction.png
├── 05_mobilesam_prediction.png
├── 06_sam21_tiny_prediction.png
├── 07_mobilesamv2_prediction.png
│
└── [运行附加代码后生成:]
    ├── 00_comparison.png          [7张图排成一行]
    └── statistics.txt             [覆盖率统计]
```

---

## 🔄 修改参数（如需选择其他图像）

要推理不同的图像，在Cell中修改：

```python
# 原来:
selected_image = all_images[0]  # 第1张

# 改为:
selected_image = all_images[5]  # 第6张
selected_image = all_images[10] # 第11张
# 等等 (同一数据集中有67张图, 索引范围 0-66)
```

---

## 📝 完整Notebook执行流程

1. ✅ Cell 1: 挂载Drive
2. ✅ Cell 2: 安装依赖
3. ✅ Cell 3: 设置路径
4. ✅ Cell 4-9: 加载6个模型 (`models_dict` 初始化)
5. ✅ Cell 10: **新添加** → 紫金山单张推理
6. ✅ Cell 11 (可选): 附加代码 → 可视化或统计

---

## 💡 常见问题

**Q: 能否推理多张图像？**

A: 可以！修改Step 1的代码将单张推理改为循环：

```python
selected_images = all_images[0:5]  # 推理前5张

for idx, selected_image in enumerate(selected_images):
    output_dir_idx = os.path.join(DRIVE, f"output_zjs_{idx}")
    # ... 将下面的代码复制进来
```

**Q: 能否使用Potsdam或Vaihingen的权重？**

A: 可以！在Step 2中修改：

```python
# 原来:
model = models_dict['LoveDA'].get(m_name)

# 改为:
model = models_dict['Potsdam'].get(m_name)   # 或 'Vaihingen'
```

**Q: 推理时间多久？**

A: 
- 使用GPU (T4): ~8-10秒
- 使用GPU (A100): ~3-5秒
- 使用CPU: ~60-80秒（不推荐）

**Q: 能否保存为更高分辨率？**

A: 可以！在Step 3前添加：

```python
# 保存为原始分辨率 (1024×1024)
pred_rgb_1024 = cv2.resize(pred_rgb, (1024, 1024))
Image.fromarray(pred_rgb_1024).save(...)
```

---

## ✨ 运行成功标志

- ✅ 控制台无红色错误信息
- ✅ 看到"✅ 推理完成！"的消息
- ✅ `output_zjs/` 目录中有7张图
- ✅ 文件大小约 100KB-500KB (取决于内容)

---

**准备好了？复制代码，粘贴到Colab，运行！** 🚀
