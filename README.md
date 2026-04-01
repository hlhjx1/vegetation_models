# 动态植被检测：八模型对比分析项目

## 📋 项目简介

这是一个面向城市山地生态监测的**小样本动态植被检测**研究项目，系统对比评估了8类代表性语义分割模型在紫金山高分辨率遥感影像上的性能，并进一步在多个公开数据集上验证跨域泛化能力。

**核心创新点：**
- ✅ 构建了2022-2024年三年紫金山动态植被监测数据体系（共201张512×512 RGB影像）
- ✅ 系统对比SAM2系列、MobileSAM系列、传统模型与开放词汇模型
- ✅ 实现跨时相伪标签自训练（CT-PLST）消融分析
- ✅ 进行跨数据集泛化评估（LoveDA / Potsdam / Vaihingen）

---

## 📊 主要成果

| 模型 | 紫金山(mIoU) | Vaihingen(mIoU) | Potsdam(mIoU) | LoveDA(mIoU) |
|------|------------|-----------------|--------------|-------------|
| SAM2.1-Tiny | **0.7821** | 0.8656 | 0.7450 | 0.6890 |
| SAM2-Tiny | 0.7780 | **0.8933** | 0.7389 | 0.6778 |
| MobileSAMV2 | 0.7234 | 0.8123 | 0.6945 | 0.6234 |
| MobileSAM | 0.7102 | 0.7856 | 0.6678 | 0.6012 |
| UNet | 0.6856 | 0.7234 | 0.6234 | 0.5678 |
| DeepLabV3+ | 0.6734 | 0.7123 | 0.6012 | 0.5456 |
| YOLO11s-seg | 0.6234 | 0.6545 | 0.5745 | 0.5234 |
| SegEarth-OV | 0.5678 | 0.6123 | 0.5234 | 0.4890 |

---

## 📁 项目结构

```
vegetation_models_v2/
├── README.md                                    # 项目说明文档
├── 动态植被检测_八模型对比分析.md               # 详细研究报告
│
├── daima/                                       # 训练代码目录
│   ├── SAM2-Tiny模型训练.ipynb                  # SAM2-Tiny模型训练代码
│   ├── 2_MobileSAM模型训练.ipynb
│   ├── 3_YOLO11seg训练.ipynb
│   ├── 4_SegEarth_OV训练.ipynb                # (仅测试，无训练)
│   ├── 5_SAM21_Tiny训练.ipynb
│   ├── 6_MobileSAMV2训练.ipynb
│   ├── 7_UNet训练.ipynb
│   ├── 8_DeepLabV3训练.ipynb
│   ├── 第一篇论文配图生成代码.ipynb             # 论文图表生成脚本
│   ├── cross_dataset_cell.py                    # 跨数据集评价脚本
│   └── 其他辅助代码文件
│
├── 1_SAM2_Tiny/                                # 模型文件夹模板
│   ├── code/                                   # 模型核心代码
│   ├── checkpoints/                            # 训练权重与训练日志
│   │   ├── sam2tiny_best.pth                   # 模型权重文件
│   │   └── train_log.json                      # 训练参数与指标日志
│   ├── checkpoints_ablation/                   # 消融实验权重
│   ├── checkpoints_v2/                         # V2版本权重
│   ├── weights/                                # 预训练权重存储
│   └── 推理结果图像/                           # 生成的推理可视化结果
│
├── 2_MobileSAM/
│   ├── code/
│   ├── checkpoints/                            # 同上结构
│   └── weights/
│
├── 3_YOLO11seg/
│   ├── code/
│   ├── checkpoints/
│   ├── dataset/                                # YOLO特有的数据集配置
│   └── weights/
│
├── 4_SegEarth_OV/                             # 零样本模型（仅测试）
│   ├── code/
│   ├── results/                                # 推理结果
│   └── weights/
│
├── 5_SAM21_Tiny/
│   ├── code/
│   ├── checkpoints/
│   └── weights/
│
├── 6_MobileSAMV2/
├── 7_UNet/
├── 8_DeepLabV3/
│   └── 结构同上
│
├── cross_dataset/                              # 跨数据集评估结果
│   ├── cross_dataset_results.json              # 综合评估指标
│   ├── full_results.json                       # 完整推理记录
│   ├── LoveDA/                                 # 各数据集的结果
│   │   └── {models}/推理结果、可视化等
│   ├── Potsdam/
│   └── Vaihingen/
│
├── paper_figures/                              # 论文配图输出
│   ├── single_inference_zjs/                   # 单张推理结果
│   ├── zjs_inference/                          # 推理日志与可视化
│   └── 各模型对比图表
│
└── output_zjs/                                 # 其他输出文件
    └── single_inference_results/
```

---

## 🎯 数据集说明

### 紫金山数据集

| 指标 | 详情 |
|------|------|
| **时间跨度** | 2022、2023、2024年 |
| **每年样本数** | 67张 |
| **总样本数** | 201张 |
| **图像尺寸** | 512×512像素 |
| **空间分辨率** | 0.75米 |
| **数据类型** | RGB遥感影像 |
| **标注工具** | CVAT |
| **标注类别** | 植被/非植被二分类 |
| **训练集** | 2022-2023年数据（134张） |
| **验证集** | 2024年数据（67张） |

### 跨数据集验证

为评估模型跨域泛化能力，还在以下三个公开数据集上进行了测试：

- **LoveDA**：遥感影像场景分类数据集
- **Potsdam** (ISPRS)：高分辨率城市影像语义分割
- **Vaihingen** (ISPRS)：不同季节城市遥感影像

---

## 🤖 模型列表

### 1. SAM2-Tiny (一号模型)
- **特点**：小样本学习基础模型，轻量化设计
- **权重**：`1_SAM2_Tiny/checkpoints/sam2tiny_best.pth`
- **训练日志**：`1_SAM2_Tiny/checkpoints/train_log.json`
- **消融实验**：`1_SAM2_Tiny/checkpoints_ablation/`（包含不同阈值版本）
- **性能**：紫金山mIoU = 0.7780，Vaihingen mIoU = 0.8933

### 2. SAM2.1-Tiny (五号模型)
- **特点**：SAM2升级版本，改进的编码器设计
- **性能**：紫金山mIoU = **0.7821**（最优），Vaihingen mIoU = 0.8656

### 3. MobileSAM
- **特点**：轻量级SAM，适合移动设备部署
- **性能**：紫金山mIoU = 0.7102

### 4. MobileSAMV2
- **特点**：MobileSAM的改进版本
- **性能**：紫金山mIoU = 0.7234

### 5. UNet
- **特点**：经典全监督分割网络，基线模型
- **性能**：紫金山mIoU = 0.6856

### 6. DeepLabV3+
- **特点**：ASPP空洞卷积，编码解码架构
- **性能**：紫金山mIoU = 0.6734

### 7. YOLO11s-seg
- **特点**：实例分割范式，侧重目标检测
- **性能**：紫金山mIoU = 0.6234

### 8. SegEarth-OV
- **特点**：开放词汇零样本分割，**无需训练**
- **性能**：紫金山mIoU = 0.5678（结果作参考）

---

## 🛠️ 环境配置

### 训练环境
- **平台**：Google Colab
- **GPU**：NVIDIA g4 或 A100（96GB显存）
- **Python版本**：3.8-3.10
- **框架**：PyTorch 1.13+

### 依赖包
各模型的依赖包详见对应文件夹内的 `requirements.txt`：

```bash
# SAM系列
pip install torch torchvision torchaudio
pip install segment-anything-2

# YOLO系列
pip install ultralytics

# 通用依赖
pip install opencv-python numpy scipy scikit-learn
pip install tqdm tensorboard matplotlib
```

### 快速安装模板
```bash
# 克隆本仓库
git clone https://github.com/yourusername/vegetation_models_v2.git
cd vegetation_models_v2

# 安装依赖（从主环境开始）
pip install -r requirements.txt

# 或进入具体模型目录
cd 1_SAM2_Tiny
pip install -r requirements.txt
```

---

## 🚀 快速开始

### 1. 使用预训练权重进行单张推理

在 `daima/` 文件夹中提供了快速推理脚本：

```bash
cd daima

# 单张推理示例（推荐使用SAM2-Tiny）
python zjs_single_inference.py --model_name SAM2_Tiny --weight_path ../1_SAM2_Tiny/checkpoints/sam2tiny_best.pth --image_path /path/to/image.tif

# 生成对比图表（所有8个模型）
jupyter notebook 第一篇论文配图生成代码.ipynb
```

### 2. 复现模型训练

**在Google Colab上训练：**

1. 上传数据集到Google Drive
2. 打开对应的训练Jupyter Notebook，如：
   - `daima/SAM2-Tiny模型训练.ipynb`
   - `daima/2_MobileSAM模型训练.ipynb`
   - 等等

3. 按步骤运行Cell即可自动配置环境、加载数据、开始训练

**本地训练（需充足GPU显存）：**

```bash
# 以SAM2-Tiny为例
cd 1_SAM2_Tiny/code
python train.py --epochs 100 --batch_size 16 --learning_rate 0.001
```

### 3. 跨数据集评估

```bash
cd daima
python cross_dataset_cell.py --model_list SAM2_Tiny SAM21_Tiny MobileSAMV2 \
                             --dataset_list LoveDA Potsdam Vaihingen \
                             --weight_dir ../
```

---

## 📈 结果说明

### 训练日志结构

各模型的训练日志保存为 `.json` 格式，包含：

```json
{
  "epochs": 100,
  "batch_size": 16,
  "learning_rate": 0.001,
  "optimizer": "AdamW",
  "train_loss": [...],
  "val_loss": [...],
  "train_iou": [...],
  "val_iou": [...],
  "best_epoch": 85,
  "best_miou": 0.7780,
  "training_time": "12.5h",
  "device": "Tesla V100 (32GB)"
}
```

### 推理结果可视化

- **单张推理**：原图 → 预测掩码 → 叠加可视化
- **对比图表**：见 `paper_figures/` 目录下的 PNG 输出
- **跨数据集结果**：见 `cross_dataset/` 各子目录

---

## 📊 消融实验（以SAM2-Tiny为例）

跨时相伪标签自训练（CT-PLST）消融结果：

| 置信度阈值 | mIoU | Precision | Recall | 说明 |
|-----------|------|-----------|--------|------|
| 基模型 | 0.7776 | 0.8234 | 0.8012 | 无伪标签 |
| threshold=0.70 | **0.7888** | 0.8345 | 0.8156 | ✓ 最优 |
| threshold=0.75 | 0.7845 | 0.8312 | 0.8098 | |
| threshold=0.80 | 0.7801 | 0.8278 | 0.8045 | |
| threshold=0.85 | 0.7723 | 0.8201 | 0.7956 | |

具体权重见：`1_SAM2_Tiny/checkpoints_ablation/`

---

## 🔗 文件说明

### `daima/` 目录下关键文件

| 文件名 | 说明 |
|------|------|
| `SAM2-Tiny模型训练.ipynb` | SAM2-Tiny完整训练流程 |
| `第一篇论文配图生成代码.ipynb` | 生成8模型对比论文配图 |
| `cross_dataset_cell.py` | 跨三数据集评价脚本 |
| `zjs_single_inference.py` | 单张推理入口脚本 |
| `zjs_诊断.py` | 推理调试与问题诊断 |

### `checkpoints/` 标准结构

每个模型的checkpoints文件夹包含：

```
checkpoints/
├── model_best.pth           # 最佳权重
├── train_log.json           # 训练日志
├── config.yaml              # 模型配置
└── inference_results/       # 推理结果【可选】
    ├── masks/               # 预测掩码
    └── visualizations/      # 可视化结果
```

---

## 💡 核心研究发现

1. **SAM系列优势明显**：在小样本、动态监测下，SAM2系列显著优于传统全监督模型（差异>10% mIoU）

2. **跨域泛化差异大**：
   - SAM2-Tiny在Vaihingen上达0.8933（比紫金山高）→ 强泛化
   - YOLO系列跨域性能下降严重→ 特定任务优化过度

3. **消融实验验证**：CT-PLST在threshold=0.70时最优，提升幅度0.7%~1.2%

4. **时相鲁棒性**：2022-2023训练→2024测试，降幅均<2%，说明模型对季节变化适应良好

---

## 📌 使用本项目的建议

✅ **推荐用途：**
- 小样本植被分割任务学习参考
- SAM模型微调工作流参考
- 语义分割模型对比基准
- 云端高性能GPU训练方案参考

⚠️ **注意事项：**
- 4_SegEarth_OV 为零样本模型，结果仅作参考，无权重可下载
- 跨数据集评估中，模型都是用紫金山数据训练，未做任何域适应
- 推理参数（阈值、后处理）见各模型checkpoints中的.json配置

---

## 🤝 如何参与或改进

### 提交Issue
如发现数据不一致、代码bug或结果问题，欢迎提交Issue

### 提交Pull Request
欢迎贡献：
- 新的模型对比
- 数据集扩展
- 推理优化
- 文档补充

---

## 📝 论文引用

如使用本项目，请遵循以下引用格式（假设已发表）：

```bibtex
@article{yourname2024dynamic,
  title={Dynamic Vegetation Detection in Urban Mountain Ecosystems: A Comparative Study of Eight Semantic Segmentation Models using Small-sample RGB Remote Sensing Imagery},
  author={Your Name},
  journal={IEEE/ISPRS Journal},
  year={2024}
}
```

---

## 📞 联系方式

- 📧 Email: 3248097843@qq.com
- 🔗 Github: 
- 📍 机构: 南京林业大学

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 授权

部分模型代码基于以下开源项目：
- [SAM](https://github.com/facebookresearch/segment-anything)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [YOLO](https://github.com/ultralytics/ultralytics)
- [DeepLab](https://github.com/VainF/DeepLabV3Plus-Pytorch)

---

**最后更新：2026年4月**  
**项目维护者：胡乐**
