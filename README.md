# 紫金山高分辨率遥感植被分割数据集与多模型对比

本仓库公开南京紫金山高分辨率遥感植被分割研究中的数据集说明、训练与评估代码、模型对比结果和可复现实验材料。研究以南京紫金山及周边城郊交错区域为核心场景，围绕 2022-2024 年多时相 RGB 遥感影像，构建小样本植被语义分割数据集，并系统比较多类深度学习模型在植被提取任务中的表现。

公开数据集已上传至 Hugging Face：

```text
https://huggingface.co/datasets/ledemo/zijinshan-vegetation-segmentation
```

相关论文题目：

```text
Comparative Performance Analysis of Mainstream Deep Learning and Vision Foundation Models
for Small-Sample Vegetation Segmentation in High-Resolution Remote Sensing Imagery
```

## 项目概览

本项目主要包含三部分：

- 自建紫金山植被分割数据集：201 张 512 x 512 RGB 影像及人工标注，另含 67 张伪标签样本用于 CT-PLST。
- 八模型主对比实验：SAM2-Tiny、SAM2.1-Tiny、MobileSAM、MobileSAMV2、U-Net、DeepLabV3+、YOLO11s-seg、SegEarth-OV。
- 补充实验材料：DINOv2、DINOv3、QwenVL、LocateAnything、效率测试、扰动鲁棒性评估和四数据集验证脚本。

核心研究问题包括：

- 在小样本高分辨率遥感场景下，视觉基础模型是否优于经典全监督模型。
- 跨时相伪标签自训练 CT-PLST 是否能缓解人工标注不足。
- 模型在 LoveDA、Potsdam、Vaihingen 等公开遥感数据集上的跨域稳定性如何。
- 补充基础模型与提示式模型是否适合连续植被语义区域提取。

## 数据集

紫金山数据集为作者使用 CVAT 自行标注的二分类语义分割数据集，类别为 `background` 和 `vegetation`。

| 项目 | 内容 |
| --- | --- |
| 研究区 | 南京紫金山风景区及周边城郊交错区域 |
| 时间范围 | 2022、2023、2024 |
| 空间分辨率 | 0.75 m/pixel |
| 图像类型 | RGB 遥感影像 |
| 图像尺寸 | 512 x 512 pixels |
| 人工标注样本 | 201 张 |
| 伪标签样本 | 67 张 |
| 标注工具 | CVAT |
| 公开地址 | https://huggingface.co/datasets/ledemo/zijinshan-vegetation-segmentation |

数据目录采用 Pascal VOC 风格：

```text
datasets/
  2022-seg/
  2023-seg/
  2024-seg/
  2024-pseudo/
  processed/
  README.md
```

公开数据集的详细字段、标签编码、许可和引用方式见 `datasets/README.md` 或 Hugging Face Dataset Card。

## 主要结果

### 紫金山主测试集

以下结果来自论文实验中的统一测试协议。

| Model | mIoU | F1 Score | Acc (%) | Best Epoch |
| --- | ---: | ---: | ---: | ---: |
| SAM2.1-Tiny | **0.7821** | **0.8688** | **88.70** | 3 |
| SAM2-Tiny | 0.7780 | 0.8660 | 88.33 | 6 |
| MobileSAMV2 | 0.7382 | 0.8316 | 87.28 | 16 |
| MobileSAM | 0.7369 | 0.8322 | 85.95 | 4 |
| DeepLabV3+ | 0.7345 | 0.8360 | 84.73 | 5 |
| U-Net | 0.7157 | 0.8239 | 83.48 | 10 |
| SegEarth-OV | 0.3079 | 0.4205 | 48.50 | - |
| YOLO11s-seg | 0.0670 | - | - | - |

## 原始八模型实验

原始主实验的主要训练、评估和制图入口集中在 `daima/` 文件夹。`1_SAM2_Tiny/` 到 `8_DeepLabV3/` 主要保存各模型的上游代码、权重、日志、检查点和模型相关资源。八模型实验是本项目的主体，补充实验只是在此基础上进一步扩展。

```text
daima/
  SAM2-Tiny模型训练.ipynb
  2_MobileSAM模型训练.ipynb
  3_YOLO11seg训练.ipynb
  4_SegEarth_OV训练.ipynb
  5_SAM21_Tiny训练.ipynb
  6_MobileSAMV2训练.ipynb
  7_UNet训练.ipynb
  8_DeepLabV3训练.ipynb
  对比实验_三数据集_8模型训练.ipynb
  第一篇论文配图生成代码.ipynb
  快速开始演示脚本.py
  模型训练推理结果转换脚本.py
  COLAB_快速启动.md
  ⭐_开始使用.md
  paper_figures_en/
```

### `daima/` 中的主要文件

| 文件 | 作用 |
| --- | --- |
| `SAM2-Tiny模型训练.ipynb` | SAM2-Tiny 在紫金山数据集上的训练和测试流程 |
| `2_MobileSAM模型训练.ipynb` | MobileSAM 训练和评估流程 |
| `3_YOLO11seg训练.ipynb` | YOLO11s-seg 实例分割基线训练和评估流程 |
| `4_SegEarth_OV训练.ipynb` | SegEarth-OV 开放词汇/零样本推理实验 |
| `5_SAM21_Tiny训练.ipynb` | SAM2.1-Tiny 训练和评估流程 |
| `6_MobileSAMV2训练.ipynb` | MobileSAMV2 训练和评估流程 |
| `7_UNet训练.ipynb` | U-Net 全监督语义分割基线 |
| `8_DeepLabV3训练.ipynb` | DeepLabV3+ 全监督语义分割基线 |
| `对比实验_三数据集_8模型训练.ipynb` | LoveDA / Potsdam / Vaihingen 跨数据集八模型对比实验 |
| `第一篇论文配图生成代码.ipynb` | 论文主图、对比图和结果图表生成 |
| `快速开始演示脚本.py` | 快速检查和演示入口 |
| `模型训练推理结果转换脚本.py` | 训练/推理结果整理与格式转换 |
| `COLAB_快速启动.md`、`⭐_开始使用.md` | Colab 环境和使用说明 |

### 模型资源目录

| Model family | Resource directory | Role in the study |
| --- | --- | --- |
| SAM2-Tiny | `1_SAM2_Tiny/` | Main fine-tuned vision foundation baseline and CT-PLST self-training backbone |
| MobileSAM | `2_MobileSAM/` | Lightweight SAM-family comparison model |
| YOLO11s-seg | `3_YOLO11seg/` | Instance segmentation paradigm comparison |
| SegEarth-OV | `4_SegEarth_OV/` | Open-vocabulary / zero-shot segmentation comparison |
| SAM2.1-Tiny | `5_SAM21_Tiny/` | Best-performing SAM-family model in the main experiment |
| MobileSAMV2 | `6_MobileSAMV2/` | Updated lightweight SAM-family comparison model |
| U-Net | `7_UNet/` | Classic fully supervised encoder-decoder baseline |
| DeepLabV3+ | `8_DeepLabV3/` | Classic fully supervised semantic segmentation baseline |

### Main Experiment Workflow

The original experiment follows this workflow:

1. Prepare the Zijinshan binary segmentation dataset in Pascal VOC-like format.
2. Train or fine-tune the eight models under a unified small-sample setting.
3. Evaluate all models on the Zijinshan 2024 test subset.
4. Transfer the trained models to LoveDA, Potsdam, and Vaihingen for cross-dataset testing.
5. Apply CT-PLST pseudo-label self-training to test whether cross-temporal unlabeled imagery improves the small-sample setting.
6. Generate manuscript figures and summary tables from saved logs and prediction masks.

主实验相关材料：

```text
daima/
cross_dataset/
paper_figures/
results/
```

复现时优先从 `daima/` 中的 notebook 或说明文档开始；如果 notebook 需要导入模型源码、预训练权重或检查点，再进入 `1_SAM2_Tiny/` 到 `8_DeepLabV3/` 对应目录查看模型资源。

### 跨数据集泛化

模型在紫金山数据上训练后，迁移到 LoveDA、Potsdam、Vaihingen 进行测试。

| Model | LoveDA | Potsdam | Vaihingen | Average |
| --- | ---: | ---: | ---: | ---: |
| SAM2-Tiny | 0.2906 | 0.8410 | **0.8933** | **0.6750** |
| SAM2.1-Tiny | 0.2889 | 0.8387 | 0.8868 | 0.6715 |
| MobileSAM | 0.2924 | 0.8348 | 0.8620 | 0.6631 |
| MobileSAMV2 | **0.2934** | 0.8377 | 0.8656 | 0.6656 |
| U-Net | 0.2738 | 0.8494 | 0.8611 | 0.6614 |
| DeepLabV3+ | 0.2650 | **0.8506** | 0.8598 | 0.6585 |
| YOLO11s-seg | 0.2460 | 0.5775 | 0.7663 | 0.5299 |
| SegEarth-OV | 0.0401 | 0.1820 | 0.4450 | 0.2224 |

注意：LoveDA 与紫金山数据存在明显空间分辨率差异，因此该结果应理解为域偏移与尺度差异共同作用下的鲁棒性评估，而不是严格的单因素分辨率归一化实验。

### CT-PLST 伪标签自训练

跨时相伪标签自训练 CT-PLST 用于利用 2024 年未标注或伪标注影像，缓解小样本监督不足。论文实验中报告的最佳设置为：

| Setting | Retained pseudo labels | mIoU | F1 | Acc |
| --- | ---: | ---: | ---: | ---: |
| SAM2-Tiny baseline | - | 0.7776 | - | - |
| CT-PLST, tau = 0.70 | 67 | **0.7888** | **0.8745** | **0.8877** |
| CT-PLST, tau = 0.75 | 67 | 0.7859 | - | - |
| CT-PLST, tau = 0.80 | 67 | 0.7858 | - | - |
| CT-PLST, tau = 0.85 | 66 | 0.7859 | - | - |

## 补充实验

补充实验代码集中在：

```text
新增实验相关代码_GitHub上传版/
```

该目录包含基础模型、提示式模型、效率和鲁棒性补充实验，可用于扩展主实验结果和复现实验表格。

```text
新增实验相关代码_GitHub上传版/
  9_DINOv2/              # DINOv2 四数据集 frozen-backbone segmentation 实验
  10_DINOv3/             # DINOv3 四数据集 frozen-backbone segmentation 实验
  11_QwenVL/             # QwenVL 植被提示式推理原始输出
  12_LocatingAnything/   # LocateAnything 植被 grounding 原始输出
  scripts/               # 数据准备、训练、效率、扰动鲁棒性、提示式评估脚本
  results/               # CSV / JSON / Markdown 汇总结果
  README.md              # 补充实验目录说明
```

### DINOv2 / DINOv3 四数据集结果

| Model | Dataset | Eval samples | Input size | mIoU | Veg IoU | F1 | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 ViT-L/14 reg | Zijinshan | 67 | 518 | 0.7466 | 0.8161 | 0.8987 | 0.8673 |
| DINOv2 ViT-L/14 reg | LoveDA | 100 | 518 | 0.7076 | 0.4839 | 0.6522 | 0.9355 |
| DINOv2 ViT-L/14 reg | Potsdam | 80 | 518 | 0.8403 | 0.8633 | 0.9267 | 0.9152 |
| DINOv2 ViT-L/14 reg | Vaihingen | 40 | 518 | 0.8782 | 0.8616 | 0.9256 | 0.9364 |
| DINOv3 ViT-L/16 SAT-493M | Zijinshan | 67 | 512 | 0.7683 | 0.8379 | 0.9118 | 0.8822 |
| DINOv3 ViT-L/16 SAT-493M | LoveDA | 100 | 512 | 0.7505 | 0.5621 | 0.7197 | 0.9433 |
| DINOv3 ViT-L/16 SAT-493M | Potsdam | 80 | 512 | 0.8530 | 0.8757 | 0.9337 | 0.9227 |
| DINOv3 ViT-L/16 SAT-493M | Vaihingen | 40 | 512 | 0.8793 | 0.8619 | 0.9258 | 0.9372 |

对应结果文件：

```text
新增实验相关代码_GitHub上传版/results/dinov2_four_dataset_results.csv
新增实验相关代码_GitHub上传版/results/dinov3_four_dataset_results.csv
```

### QwenVL / LocateAnything 提示式评估

QwenVL2.5-3B 和 LocateAnything-3B 不是密集语义分割训练模型。本项目仅将其预测框转为粗粒度二值 mask，用于补充提示式视觉模型在植被定位任务上的表现。

| Model | Dataset | Box-mask IoU | Box-mask F1 | Box-mask Acc |
| --- | --- | ---: | ---: | ---: |
| QwenVL2.5-3B | Zijinshan | 0.6212 | 0.6981 | 0.7557 |
| QwenVL2.5-3B | LoveDA | 0.1175 | 0.1806 | 0.6437 |
| QwenVL2.5-3B | Potsdam | 0.3482 | 0.4233 | 0.6145 |
| QwenVL2.5-3B | Vaihingen | 0.4615 | 0.6080 | 0.5290 |
| LocateAnything-3B | Zijinshan | 0.5747 | 0.6457 | 0.7055 |
| LocateAnything-3B | LoveDA | 0.1122 | 0.1756 | 0.6649 |
| LocateAnything-3B | Potsdam | 0.5038 | 0.5914 | 0.6932 |
| LocateAnything-3B | Vaihingen | 0.5444 | 0.6924 | 0.7046 |

对应结果文件：

```text
新增实验相关代码_GitHub上传版/results/qwenvl_prompt_eval_results.csv
新增实验相关代码_GitHub上传版/results/locateanything_grounding_results.csv
新增实验相关代码_GitHub上传版/results/locateanything_grounding_summary.md
```

### 效率与鲁棒性补充

补充实验中还加入了模型效率和扰动鲁棒性评估，主要脚本包括：

```text
新增实验相关代码_GitHub上传版/scripts/benchmark_model_efficiency.py
新增实验相关代码_GitHub上传版/scripts/run_robustness_perturbation_eval.py
新增实验相关代码_GitHub上传版/scripts/colab_four_model_efficiency_robustness.py
新增实验相关代码_GitHub上传版/scripts/run_dinov2_four_dataset_train.py
新增实验相关代码_GitHub上传版/scripts/run_dinov3_four_dataset_train.py
新增实验相关代码_GitHub上传版/scripts/run_qwenvl_vegetation_prompt_eval.py
新增实验相关代码_GitHub上传版/scripts/run_locateanything_vegetation_grounding.py
```

汇总结果可见：

```text
新增实验相关代码_GitHub上传版/results/model_efficiency_summary.md
新增实验相关代码_GitHub上传版/results/robustness_perturbation_summary.md
新增实验相关代码_GitHub上传版/results/four_model_robustness_colab.csv
新增实验相关代码_GitHub上传版/results/four_model_efficiency_colab.csv
```

## 仓库结构

```text
vegetation_models_v2/
  README.md
  datasets/                         # 自建紫金山数据集说明和本地数据
  daima/                            # 原始八模型训练、跨数据集实验和论文配图核心代码
  1_SAM2_Tiny/
  2_MobileSAM/
  3_YOLO11seg/
  4_SegEarth_OV/
  5_SAM21_Tiny/
  6_MobileSAMV2/
  7_UNet/
  8_DeepLabV3/
  cross_dataset/                    # 原始跨数据集评估相关内容
  新增实验相关代码_GitHub上传版/       # 补充实验公开版
```

## 快速开始

### 1. 获取数据集

推荐从 Hugging Face 下载公开数据集：

```bash
git lfs install
git clone https://huggingface.co/datasets/ledemo/zijinshan-vegetation-segmentation
```

也可以在 Python 中使用 `huggingface_hub` 下载：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ledemo/zijinshan-vegetation-segmentation",
    repo_type="dataset",
    local_dir="datasets"
)
```

### 2. 安装依赖

不同模型依赖差异较大，建议进入对应模型目录或脚本目录后按需安装。通用环境可从以下包开始：

```bash
pip install torch torchvision torchaudio
pip install numpy pandas pillow opencv-python scikit-learn matplotlib tqdm
pip install huggingface_hub
```

部分模型还需要各自官方仓库或权重：

- SAM2 / SAM2.1：Meta SAM2 相关依赖和 Hiera 权重。
- MobileSAM / MobileSAMV2：对应官方实现及 TinyViT 权重。
- YOLO11s-seg：Ultralytics。
- SegEarth-OV：CLIP / open-clip / SimFeatUp 相关依赖。
- DINOv2 / DINOv3：对应 backbone 权重和本仓库训练得到的 segmentation head。
- QwenVL / LocateAnything：Hugging Face 模型权重和 `transformers` 相关依赖。

### 3. 运行原始八模型实验

原始实验优先从 `daima/` 目录运行。该目录包含八个模型的训练 notebook、三数据集八模型对比 notebook、结果转换脚本和论文配图生成代码。

```bash
cd daima
```

主要入口：

```text
SAM2-Tiny模型训练.ipynb
2_MobileSAM模型训练.ipynb
3_YOLO11seg训练.ipynb
4_SegEarth_OV训练.ipynb
5_SAM21_Tiny训练.ipynb
6_MobileSAMV2训练.ipynb
7_UNet训练.ipynb
8_DeepLabV3训练.ipynb
对比实验_三数据集_8模型训练.ipynb
第一篇论文配图生成代码.ipynb
```

快速检查或辅助脚本：

```bash
python 快速开始演示脚本.py
python 模型训练推理结果转换脚本.py
```

如果 notebook 需要模型源码、权重或检查点，再进入 `1_SAM2_Tiny/` 到 `8_DeepLabV3/` 的对应模型资源目录。

### 4. 运行补充实验脚本

示例：

```bash
cd 新增实验相关代码_GitHub上传版

# DINOv2 四数据集训练/验证
python scripts/run_dinov2_four_dataset_train.py

# DINOv3 四数据集训练/验证
python scripts/run_dinov3_four_dataset_train.py

# QwenVL 提示式植被框评估
python scripts/run_qwenvl_vegetation_prompt_eval.py

# LocateAnything grounding 评估
python scripts/run_locateanything_vegetation_grounding.py
```

具体路径可能需要根据本地数据集和权重位置调整，建议先阅读脚本顶部参数和 `新增实验相关代码_GitHub上传版/README.md`。

## 复现说明

- 本项目包含多类模型和多个第三方代码源，不同模型的依赖并不完全一致。
- 大型预训练基础模型权重通常不随 GitHub 仓库直接提供，请按对应官方许可自行下载。
- 自建紫金山数据集已公开在 Hugging Face；公开仓库中若不包含完整数据，请以 Hugging Face 数据集为准。
- QwenVL 和 LocateAnything 的结果属于提示式/grounding 补充实验，不应与密集语义分割模型的 mIoU 直接混为同一排行榜。

## 引用

如果使用本项目代码或数据集，请引用相关论文和数据集。论文正式发表或 DOI 更新后，请替换以下占位信息。

```bibtex
@article{hu2026zijinshan_vegetation_segmentation,
  title={Comparative Performance Analysis of Mainstream Deep Learning and Vision Foundation Models for Small-Sample Vegetation Segmentation in High-Resolution Remote Sensing Imagery},
  author={Hu, Le and Zhang, Fuquan},
  journal={Remote Sensing},
  year={2026},
  note={Manuscript under review}
}

@dataset{hu2026zijinshan_dataset,
  title={Zijinshan Vegetation Segmentation Dataset},
  author={Hu, Le and Zhang, Fuquan},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/ledemo/zijinshan-vegetation-segmentation}
}
```

## 许可

- 代码部分建议按仓库中的开源许可证执行。
- 紫金山数据集在 Hugging Face 上按 `CC BY-NC 4.0` 发布，具体以数据集页面为准。
- 第三方模型、预训练权重和外部数据集请遵循其原始许可证和使用条款。

## 联系方式

- 作者：Hu Le
- 单位：南京林业大学，信息科学技术学院、人工智能学院
- 邮箱：3248097843@qq.com
- 数据集：`ledemo/zijinshan-vegetation-segmentation`
