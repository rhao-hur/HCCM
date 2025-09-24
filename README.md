# HCCM: Hierarchical Cross-Granularity Contrastive and Matching Learning for Natural Language-Guided Drones

[![Paper](https://img.shields.io/badge/arXiv-2508.21539-b31b1b.svg)](https://arxiv.org/pdf/2508.21539)
[![Pretrained Model](https://img.shields.io/badge/Model-Download-blue.svg)](https://drive.google.com/file/d/1p468glkjTqxuE7YhzdXnC1Kx4xEJQEM3/view?usp=sharing)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

**这是我们发表于 ACM MM 2025 的论文 "HCCM" 的官方代码实现。**

## 动态与更新 (News & Updates)

*   **[2025.09.24]** 我们正式开源了 HCCM 的完整代码以及在 GeoText1652 数据集上训练的模型权重。
*   **[2025.09.23]** 我们的方法在 [RoboSense 2025 挑战赛 - 赛道 4](https://robosense2025.github.io/track4) 中成功入围决赛前五，进入最终的代码审查阶段。该挑战赛是机器人领域顶级会议 **IROS 2025** 的官方研讨会之一。
*   **[2025.07.06]** 我们的论文 "HCCM: Hierarchical Cross-Granularity Contrastive and Matching Learning for Natural Language-Guided Drones" 已被 **ACM Multimedia (MM) 2025** 会议正式接收。

## 快速开始

首先，克隆本仓库到您的本地设备：
```bash
git clone https://github.com/rhao-hur/HCCM.git
cd HCCM
```

## 环境配置

```bash
conda create -n hccm python=3.9.20 -y
conda activate hccm
```

接下来，请严格遵循以下步骤安装所有必需的依赖库：

```bash
# 步骤 1: 安装 PyTorch (针对 CUDA 11.8 版本)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# 步骤 2: 安装基础依赖库
pip install ftfy==6.3.0 regex==2024.9.11 fsspec==2024.10.0 "protobuf>=3.2.0"
pip install opencv-python==4.10.0.84

# 步骤 3: 安装 OpenMMLab 相关库
pip install -U openmim==0.3.9 
mim install mmengine==0.10.5
pip install "numpy==1.26.4" scipy
mim install "mmcv==2.1.0"

# 步骤 4: 安装 Transformers 和 MMPretrain
pip install transformers==4.12.5 tokenizers==0.10.3
mim install "mmpretrain==1.2.0"

# 步骤 5: 安装其他必要库
pip install pycocotools==2.0.8
pip install timm==0.6.13
```
请确保您的项目根目录最后符合以下结构：
```
.
├── datasets/
│   ├── ERA_Dataset/
│   └── GeoText1652_Dataset/
│       ├── images/
│       ├── train.json
│       └── test_951_version.json
└── HCCM/
    ├── configs/
    ├── pretrain/
    │   ├── bert-base-uncased/
    │   ├── 16m_base_model_state_step_199999.th
    │   └── 16m_base_model_state_step_199999_(xvlm2mmcv).pth
    ├── src/
    ├── tools/
    ├── process_ERA_dataset.py
    ├── process_xvlm2mmcv.py
    └── README.md
```

## 数据集与模型准备

### 1. 数据集准备

首先，在项目根目录下创建 `datasets` 文件夹。

```bash
mkdir datasets
cd datasets
```

#### GeoText1652 数据集

我们推荐使用 `huggingface-cli` 从其官方仓库下载 GeoText1652 数据集。

```bash
huggingface-cli download --repo-type dataset --resume-download truemanv5666/GeoText1652_Dataset --local-dir GeoText1652_Dataset
```

下载完成后，进入 `images` 目录，解压所有的图像压缩包并删除原文件。

```bash
cd GeoText1652_Dataset/images
find . -type f -name "*.tar.gz" -print0 | xargs -0 -I {} bash -c 'tar -xzf "{}" -C "$(dirname "{}")" && rm "{}"'
cd ../..
```

#### ERA 数据集

首先，为 ERA 数据集创建相应目录。

```bash
mkdir ERA_Dataset
cd ERA_Dataset
```

ERA 数据集的图像文件 (`era_images.zip`) 可以从 [VCSR GitHub 仓库](https://github.com/huangjh98/VCSR) 获得。请下载该文件，放置于 `datasets/ERA_Dataset/` 目录下，然后解压。

```bash
unzip era_images.zip
mv era_images images
rm era_images.zip
```

接下来，我们需要生成与 GeoText1652 格式兼容的 `test.json` 标注文件。为此，我们先克隆 VCSR 仓库以获取原始数据，然后运行处理脚本。

```bash
# 克隆 VCSR 仓库以获取原始标注
git clone https://github.com/huangjh98/VCSR.git
# 运行我们的处理脚本，生成 test.json
python ../../HCCM/process_ERA_dataset.py VCSR/data/era_precomp test.json
# 操作完成后返回项目根目录
cd ../../HCCM
```

### 2. 模型准备

#### XVLM 预训练模型

请从以下链接下载 [X-VLM](https://github.com/zengyan-97/X-VLM) 的预训练模型，并将其保存到 `HCCM/pretrain` 目录中。

*   [X-VLM (4M, 200K steps)](https://drive.google.com/file/d/1B3gzyzuDN1DU0lvt2kDz2nTTwSKWqzV5/view?usp=sharing)
*   [X-VLM (16M, 200K steps)](https://drive.google.com/file/d/1iXgITaSbQ1oGPPvGaV0Hlae4QiJG5gx0/view?usp=sharing)

下载完成后，执行提供的脚本，将原始的 XVLM 模型权重转换为 MMCV 兼容的格式。

```bash
python process_xvlm2mmcv.py \
--input_path pretrain/16m_base_model_state_step_199999.th \
--output_path "pretrain/16m_base_model_state_step_199999_(xvlm2mmcv).pth"
```

#### BERT 模型

使用 `huggingface-cli` 工具将 `bert-base-uncased` 模型下载到 `HCCM/pretrain` 目录。

```bash
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir pretrain/bert-base-uncased
```

## 在 GeoText1652 数据集上训练 HCCM

默认情况下，训练脚本会自动从 `../datasets` 路径加载数据。如果您将数据集存放在了其他位置，请务必修改以下配置文件中的 `data_root` 参数：

*   `HCCM/configs/_base_/datasets/geotext1652_retrieval.py`
*   `HCCM/configs/_base_/datasets/ERA_retrieval.py`

我们所有的实验均在单张 80G A800 GPU 上完成。默认配置下，每个 epoch 会进行一次验证。请注意，我们尚未对多卡训练的代码进行充分验证。

**单卡训练:**

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/exp/xvlm_1xb24_hccm_geotext1652.py
```

**多卡训练与验证的特别说明:**

由于完整的测试集数据量较大，验证过程对 GPU 显存有较高要求。为了能够在资源受限（如 24G 显存）的设备上运行验证或测试，我们在代码中采用了将数据卸载到 CPU 的策略。这意味着，**如果训练过程中开启了周期性验证（即 `val_interval <= max_epochs`），训练将只能在单张 GPU 上进行**。

若您希望使用多卡进行训练，则必须**关闭周期性验证**。具体操作是，将 `val_interval` 的值设置为大于 `max_epochs`。例如，在 `configs/exp/xvlm_1xb24_hccm_geotext1652.py` 文件中，将 `train_cfg` 修改如下：

```python
# 将 val_interval 设置为一个大于 max_epochs 的值，以在多卡训练时禁用验证
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=10)
```

多卡训练命令（以 4 卡为例）：

```bash
bash tools/dist_train.sh configs/exp/xvlm_1xb24_hccm_geotext1652.py 4
```

## 模型测试

您可以从 [**此处**](https://drive.google.com/file/d/1p468glkjTqxuE7YhzdXnC1Kx4xEJQEM3/view?usp=sharing) 下载我们在 GeoText1652 数据集上训练了 6 个 epoch 的 HCCM 模型权重 (`epoch_6.pth`)。

### 在 GeoText1652 数据集上测试

为了完成全面的双向检索测试（Image-to-Text 和 Text-to-Image），您需要分别执行以下两步：

1.  **测试 Image-to-Text (I2T) 方向：**
    *   打开 `configs/exp/xvlm_1xb24_hccm_geotext1652.py` 文件。
    *   找到 `test_cfg` 参数，并设置 `i2t=True`。
    *   执行测试命令：
    ```bash
    CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/exp/xvlm_1xb24_hccm_geotext1652.py \
    epoch_6.pth 
    ```

2.  **测试 Text-to-Image (T2I) 方向：**
    *   再次修改 `configs/exp/xvlm_1xb24_hccm_geotext1652.py` 文件。
    *   将 `test_cfg` 参数中的 `i2t` 设置为 `False`。
    *   重新执行测试命令：
    ```bash
    CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/exp/xvlm_1xb24_hccm_geotext1652.py \
    epoch_6.pth 
    ```

### 在 ERA 数据集上进行 Zero-Shot 测试

要评估 HCCM 在 ERA 数据集上的零样本泛化能力，请运行以下命令。同样地，您需要通过修改配置文件来分别测试 I2T 和 T2I 两个方向。

*   打开 `configs/exp/xvlm_1xb24_hccm_ERAzeroshot.py` 文件。
*   根据测试需求，将 `test_cfg` 中的 `i2t` 参数设置为 `True` (I2T) 或 `False` (T2I)。
*   执行测试命令：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
configs/exp/xvlm_1xb24_hccm_ERAzeroshot.py \
epoch_6.pth 
```

## 致谢 (Acknowledgements)

我们的研究工作建立在众多优秀的开源项目之上。在此，我们向以下项目的开发者和贡献者表示最诚挚的感谢：

- [X-VLM](https://github.com/zengyan-97/X-VLM): 一个强大的多粒度视觉语言对齐预训练模型，为我们的工作提供了坚实的起点。
- [GeoText-1652](https://github.com/MultimodalGeo/GeoText-1652): 一个专注于无人机导航中空间关系匹配的高质量基准数据集。
- [OpenMMLab](https://github.com/open-mmlab): 感谢 [MMPretrain](https://github.com/open-mmlab/mmpretrain)、[MMCV](https://github.com/open-mmlab/mmcv) 和 [MMEngine](https://github.com/open-mmlab/mmengine) 提供的高效、可扩展的深度学习框架，极大地简化了我们的研发流程。