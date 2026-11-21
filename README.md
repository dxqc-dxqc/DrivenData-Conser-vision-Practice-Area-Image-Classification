# ConserVision 相机陷阱图像分类

这是我在 DrivenData ConserVision 练习赛中的图像分类实验。我按拍摄站点划分训练集和验证集，冻结 Google SpeciesNet 主干并训练八分类线性头；新方法再用 MegaDetector 定位动物、提取检测框裁剪图的 SpeciesNet 特征，并将裁剪预测与全图预测融合，识别 7 个动物类别和空镜（`blank`）。

## 实验结果

验证集含 578 张图片，按拍摄站点与训练集隔离。基线全图模型的 Log Loss 为 **0.61271**，准确率 **81.49%**。50/50 概率融合后，Log Loss 降至 **0.57864**，准确率升至 **83.04%**。MegaDetector 在 578 张验证图中检测到 407 张（覆盖率 70.42%）；未检测到动物的图片沿用全图预测。裁剪分类头单独指标只在检测到动物的样本上计算，不与全量验证结果直接比较。这些是本地验证结果，不是 DrivenData 排行榜成绩。

| 产物 | 说明 |
| --- | --- |
| [分类头权重](split_runs/run_20261120_203324_14d448/speciesnet_v4.0.3b/best_head.pt) | 自行训练的第 3 轮分类头。 |
| [验证指标](split_runs/run_20261120_203324_14d448/speciesnet_v4.0.3b/metrics.json) | 各轮指标与最佳轮混淆矩阵。 |
| [训练日志](speciesnet-training-20261120-223319.log) | 特征提取与 15 轮训练记录。 |
| [提交 CSV](submission_speciesnet_epoch3.csv) | 第 3 轮模型的测试集预测。 |
| [50/50 融合提交](submission_speciesnet_fusion_5050.csv) | 新方法对测试集的预测；裁剪概率与全图概率等权融合。 |
| [裁剪头指标](split_runs/run_20261120_203324_14d448/speciesnet_v4.0.3a_crop/metrics.json) | 训练过程及检测到动物子集上的验证指标。 |
| [融合评估](split_runs/run_20261120_203324_14d448/speciesnet_v4.0.3a_crop/fusion.json) | 全部 578 张验证图上的 50/50 融合指标。 |

## 项目文件

| 文件 | 用途 |
| --- | --- |
| [`site_splitter.py`](site_splitter.py) | 查看站点与类别分布，手动选择验证站点并导出数据划分。 |
| [`train_speciesnet.py`](train_speciesnet.py) | 冻结 SpeciesNet v4.0.3b 主干，提取 1280 维特征并训练八分类线性头。 |
| [`prepare_crop_features.py`](prepare_crop_features.py) | 用 MegaDetector 定位动物，提取 SpeciesNet 4.0.3a 检测框特征。 |
| [`train_speciesnet_crop.py`](train_speciesnet_crop.py) | 使用已提取的检测框特征训练分类头。 |
| [`evaluate_crop_full_fusion.py`](evaluate_crop_full_fusion.py) | 评估全图与裁剪预测的指定权重融合，并可导出预测 CSV。 |
| [`export_full_validation_probs.py`](export_full_validation_probs.py) | 从全图特征缓存导出验证集概率，供融合评估使用。 |
| [`requirements.txt`](requirements.txt) | Python 用户界面和裁剪流程依赖；PyTorch 按本机 CUDA 版本单独安装。 |
| [`speciesnet-guide.html`](speciesnet-guide.html) | 可在浏览器打开的实验说明。 |
| [`run_speciesnet_training.ps1`](run_speciesnet_training.ps1) | 旧实验的启动脚本，固定了本地划分路径；新实验请直接使用下文命令。 |

## 复现实验

### 1. 准备环境和数据

以下以 Windows PowerShell 为例。本地使用 Python 3.12、PyTorch 2.11.0 和 torchvision 0.26.0（CUDA 12.8）。特征提取和训练要求可用的 NVIDIA CUDA 环境；站点划分不需要 GPU。

```powershell
python -m venv .venv-cuda
$python = '.\.venv-cuda\Scripts\python.exe'
& $python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -r requirements.txt
& $python -c "import torch; print(torch.cuda.is_available())"
```

从 [DrivenData 竞赛数据页](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/data/) 自行下载数据。数据目录须包含 `train_features/` 图片目录、`train_features.csv`（至少包含 `id`、`filepath`、`site`）和 `train_labels.csv`（`id` 与八个类别列，每张图只有一个正类）。图片路径须与 `filepath` 一致。

### 2. 按站点划分

```powershell
& $python .\site_splitter.py
```

在界面中选择数据目录，勾选验证站点并生成划分。选中站点的全部图片进入验证集，其余站点进入训练集，避免同一拍摄站点同时出现在两边。结果写入数据目录的 `split_runs/run_*`，包含 `train/`、`validation/` 和 `split_summary.json`。图片优先通过硬链接导出，不支持时复制，原始数据不变。

### 3. 训练 SpeciesNet 分类头

预训练主干来自 [Google SpeciesNet 模型页](https://www.kaggle.com/models/google/speciesnet)。主干权重约 214 MiB，不随 Git 克隆下载。请从 [GitHub Release 下载权重](https://github.com/dxqc-dxqc/DrivenData-Conser-vision-Practice-Area-Image-Classification/releases/tag/speciesnet-v4.0.3b)，将 `full_image_88545560_22x8_v12_epoch_00153.pt` 放到 `models/speciesnet-v4.0.3b/`，与 `info.json` 同目录。只加载可信来源的权重文件。

将 `$split` 改为上一步导出的实际目录：

```powershell
$split = '.\split_runs\run_实际目录名'
& $python .\train_speciesnet.py --split $split --batch-size 8 --epochs 15
```

可用 `--model <目录>` 指定其他权重目录。加上 `--benchmark 128` 则只测量最多 128 张训练图片的特征提取速度，不训练分类头。

产物写入划分目录下的 `speciesnet_v4.0.3b/`：`best_head.pt` 保存按验证集 Log Loss 选出的分类头，`metrics.json` 记录各轮指标和最佳轮的混淆矩阵，`*_features_*.pt` 是可复用的特征缓存。该训练脚本不生成比赛提交 CSV。

### 4. 训练检测框裁剪分类头并融合

该流程使用 SpeciesNet **4.0.3a always-crop** 权重，以及 MegaDetector v6 YOLOv10-e 权重；两者需按各自发布页的说明取得，并分别放入 `models/speciesnet-v4.0.3a/` 和 `models/megadetector/`。默认检测器路径为 `models/megadetector/MDV6-yolov10-e-1280.pt`。权重及许可信息以各上游发布页为准。

```powershell
$split = '.\split_runs\run_实际目录名'
$crop = "$split\crop_full"
$head = "$split\speciesnet_v4.0.3a_crop"
& $python .\prepare_crop_features.py --split $split --output $crop
& $python .\train_speciesnet_crop.py --features $crop --output $head
```

全图特征缓存需先由 `train_speciesnet.py` 生成。使用 `export_full_validation_probs.py` 导出验证集全图概率后，可复核融合指标并生成提交：

```powershell
& $python .\export_full_validation_probs.py --split $split --run "$split\speciesnet_v4.0.3b" --output "$split\full_validation_probs.csv"
& $python .\evaluate_crop_full_fusion.py --features $crop --head "$head\best_head.pt" --full "$split\full_validation_probs.csv" --output "$head\fusion.json" --crop-weight 0.5 --submission .\submission_speciesnet_fusion_5050.csv
```

融合权重应在独立验证集上确定。README 所列 50/50 结果来自当前站点划分，不应视为排行榜成绩。

## 数据与许可

仓库不包含比赛原始图片、数据表、官方示例 Notebook、划分数据或特征缓存。比赛数据遵循 DrivenData 的条款。预训练 SpeciesNet 与 MegaDetector 权重均为上游项目产物，不代表本项目训练成果；使用和再分发时请遵循各上游发布页的许可。本项目尚未声明自有代码许可证。
