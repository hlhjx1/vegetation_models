# 🎯 紫金山单张推理 - 三部分独立运行指南

## 为什么分成三部分？

分开运行的好处：
1. ✅ 清理全局状态（torch.load、Hydra、sys.path）
2. ✅ 避免模块间相互干扰
3. ✅ 增强稳定性（跟您的成品方案一致）
4. ✅ 支持 Colab 分 cell 运行

---

## 📋 运行步骤

### 🟦 Part 1/3: 安装和配置（第1个Colab Cell）

文件：`zjs_推理_Part1_安装和配置.py`

**功能**：
- 导入所有依赖
- **修补 torch.load**（关键！解决递归问题）
- 配置 sys.path
- 初始化全局变量

**运行**：
```python
# 在 Colab 中运行
exec(open('/content/drive/MyDrive/vegetation_models_v2/daima/zjs_推理_Part1_安装和配置.py').read())
```

**输出**：
```
✅ torch.load 修补完成
✅ sys.path 配置完成
✅ Part 1 完成！下一步运行 Part 2
```

---

### 🟩 Part 2/3: 定义函数（第2个Colab Cell）

文件：`zjs_推理_Part2_定义函数.py`

**功能**：
- 改变工作目录（Hydra）
- 定义6个模型加载函数
- 定义推理函数

**运行**：
```python
# 在 Colab 中运行
exec(open('/content/drive/MyDrive/vegetation_models_v2/daima/zjs_推理_Part2_定义函数.py').read())
```

**输出**：
```
✅ 工作目录已改为: .../1_SAM2_Tiny/code
✅ 所有函数定义完成！
✅ Part 2 完成！下一步运行 Part 3
```

---

### 🟪 Part 3/3: 推理和保存（第3个Colab Cell）

文件：`zjs_推理_Part3_加载推理.py`

**功能**：
- 加载所有6个模型
- 选择图像并推理
- 保存结果到 `output_zjs/`

**运行**：
```python
# 在 Colab 中运行
exec(open('/content/drive/MyDrive/vegetation_models_v2/daima/zjs_推理_Part3_加载推理.py').read())
```

**输出**：
```
⏳ 加载所有6个模型...
✅ UNet加载成功
✅ DeepLabV3+加载成功
✅ SAM2_Tiny加载成功
✅ MobileSAM加载成功
✅ SAM2.1_Tiny加载成功
✅ MobileSAMV2加载成功
✅ 所有模型加载完成

⏳ Step 1/3: 选择图像...
  ✅ 找到 67 张图像
  ✅ 选中第1张: omap_2024_000000.png

⏳ Step 2/3: 模型推理...
  推理 UNet           ... ✅ (植被覆盖率: 99.84%)
  推理 DeepLabV3+     ... ✅ (植被覆盖率: 99.84%)
  推理 SAM2_Tiny      ... ✅ (植被覆盖率: 99.90%)
  推理 MobileSAM      ... ✅ (植被覆盖率: 99.92%)
  推理 SAM2.1_Tiny    ... ✅ (植被覆盖率: 100.00%)
  推理 MobileSAMV2    ... ✅ (植被覆盖率: 99.99%)

⏳ Step 3/3: 保存预测结果...
  💾 UNet            → 02_unet_prediction.png (覆盖率: 99.84%)
  💾 DeepLabV3+      → 03_deeplabv3plus_prediction.png (覆盖率: 99.84%)
  ...

✅ 推理完成！
```

---

## 📁 文件说明

```
daima/
├── zjs_推理_Part1_安装和配置.py      ← Part 1: 安装依赖和修补
├── zjs_推理_Part2_定义函数.py        ← Part 2: 定义所有模型加载函数
├── zjs_推理_Part3_加载推理.py        ← Part 3: 推理和保存结果
└── zjs_推理_README.md               ← 本文件
```

---

## ⚙️ 关键配置

### torch.load 修补（Part 1）
```python
import torch.serialization
_original_load = torch.load

def _patched_load(f, *args, **kwargs):
    """允许加载包含numpy/复杂类型的权重"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)

torch.load = _patched_load
```

**为什么需要？**
- 解决 `maximum recursion depth exceeded` 错误
- 支持加载包含 numpy 类型的权重文件

### 工作目录切换（Part 2）
```python
_sam2_code_dir = os.path.join(DRIVE, "1_SAM2_Tiny/code")
if os.path.exists(_sam2_code_dir):
    os.chdir(_sam2_code_dir)
    sys.path.insert(0, _sam2_code_dir)
```

**为什么需要？**
- 让 Hydra 能找到 `sam2/sam2_hiera_t.yaml` 配置文件
- 解决 `GlobalHydra is not initialized` 错误

---

## 💡 Colab 快速复制模板

### Cell 1：
```python
exec(open('/content/drive/MyDrive/vegetation_models_v2/daima/zjs_推理_Part1_安装和配置.py').read())
```

### Cell 2：
```python
exec(open('/content/drive/MyDrive/vegetation_models_v2/daima/zjs_推理_Part2_定义函数.py').read())
```

### Cell 3：
```python
exec(open('/content/drive/MyDrive/vegetation_models_v2/daima/zjs_推理_Part3_加载推理.py').read())
```

---

## 📊 预期输出

✅ **output_zjs/ 文件夹：**
```
output_zjs/
├── 01_original_image.png          （原始RGB图像，512×512）
├── 02_unet_prediction.png         （UNet预测，植被覆盖率 ~99.84%）
├── 03_deeplabv3plus_prediction.png（DeepLabV3+预测，植被覆盖率 ~99.84%）
├── 04_sam2_tiny_prediction.png    （SAM2_Tiny预测，植被覆盖率 ~99.90%）
├── 05_mobilesam_prediction.png    （MobileSAM预测，植被覆盖率 ~99.92%）
├── 06_sam21_tiny_prediction.png   （SAM2.1_Tiny预测，植被覆盖率 ~100.00%）
└── 07_mobilesamv2_prediction.png  （MobileSAMV2预测，植被覆盖率 ~99.99%）
```

---

## 🐛 故障排除

| 问题 | 解决方案 |
|------|--------|
| `maximum recursion depth exceeded` | 确保 Part 1 的 torch.load 修补正确执行 |
| `GlobalHydra is not initialized` | 确保 Part 2 中改变了工作目录 |
| 模型都返回 0.0% | 检查是否跳过了任何 Part，按顺序运行 |
| 权重文件找不到 | 检查你的 `BASE_DIR` 是否正确指向 vegetation_models_v2 |

---

## ✨ 关键改进

vs. 之前的一体化脚本：
- ✅ 分离 torch.load 修补（避免全局污染）
- ✅ 分离工作目录切换（Hydra 初始化更干净）
- ✅ 分离模型加载（清理全局状态）
- ✅ 完全匹配您的成品方案

---

祝运行成功！如有问题检查上表的故障排除。
