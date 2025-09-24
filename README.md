# HCCM: Hierarchical Cross-Granularity Contrastive and Matching Learning for Natural Language-Guided Drones

[![Paper](https://img.shields.io/badge/arXiv-2508.21539-b31b1b.svg)](https://arxiv.org/pdf/2508.21539)
[![Pretrained Model](https://img.shields.io/badge/Model-Download-blue.svg)](https://drive.google.com/file/d/1p468glkjTqxuE7YhzdXnC1Kx4xEJQEM3/view?usp=sharing)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

**This is the official implementation for our ACM MM 2025 paper, "HCCM".**

## News & Updates

*   **[2025.09.24]** We have officially released the complete code for HCCM and the model weights trained on the GeoText1652 dataset.
*   **[2025.09.23]** Our method has been selected as one of the top five finalists in the [RoboSense 2025 Challenge - Track 4](https://robosense2025.github.io/track4), advancing to the final code review stage. This challenge is an official workshop of the top-tier robotics conference, **IROS 2025**.
*   **[2025.07.06]** Our paper, "HCCM: Hierarchical Cross-Granularity Contrastive and Matching Learning for Natural Language-Guided Drones," has been officially accepted by **ACM Multimedia (MM) 2025**.

## Quick Start

First, clone this repository to your local machine:
```bash
git clone https://github.com/rhao-hur/HCCM.git
cd HCCM
```

## Environment Setup

```bash
conda create -n hccm python=3.9.20 -y
conda activate hccm
```

Next, please strictly follow the steps below to install all the required dependencies:

```bash
# Step 1: Install PyTorch (for CUDA 11.8)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# Step 2: Install basic dependencies
pip install ftfy==6.3.0 regex==2024.9.11 fsspec==2024.10.0 "protobuf>=3.2.0"
pip install opencv-python==4.10.0.84

# Step 3: Install OpenMMLab libraries
pip install -U openmim==0.3.9 
mim install mmengine==0.10.5
pip install "numpy==1.26.4" scipy
mim install "mmcv==2.1.0"

# Step 4: Install Transformers and MMPretrain
pip install transformers==4.12.5 tokenizers==0.10.3
mim install "mmpretrain==1.2.0"

# Step 5: Install other necessary libraries
pip install pycocotools==2.0.8
pip install timm==0.6.13
```
Please ensure your project's root directory is structured as follows:

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

## Dataset and Model Preparation

### 1. Dataset Preparation

First, create a `datasets` folder in the project root directory.

```bash
mkdir datasets
cd datasets
```

#### GeoText1652 Dataset

We recommend using `huggingface-cli` to download the GeoText1652 dataset from its official repository.

```bash
huggingface-cli download --repo-type dataset --resume-download truemanv5666/GeoText1652_Dataset --local-dir GeoText1652_Dataset
```

After the download is complete, navigate to the `images` directory, extract all the compressed image archives, and then remove the original files.

```bash
cd GeoText1652_Dataset/images
find . -type f -name "*.tar.gz" -print0 | xargs -0 -I {} bash -c 'tar -xzf "{}" -C "$(dirname "{}")" && rm "{}"'
cd ../..
```

#### ERA Dataset

First, create the corresponding directory for the ERA dataset.

```bash
mkdir ERA_Dataset
cd ERA_Dataset
```

The image files for the ERA dataset (`era_images.zip`) can be obtained from the [VCSR GitHub repository](https://github.com/huangjh98/VCSR). Please download this file, place it in the `datasets/ERA_Dataset/` directory, and then unzip it.

```bash
unzip era_images.zip
mv era_images images
rm era_images.zip
```

Next, we need to generate a `test.json` annotation file that is compatible with the GeoText1652 format. To do this, we first clone the VCSR repository to get the original data, and then run our processing script.

```bash
# Clone the VCSR repository to get the original annotations
git clone https://github.com/huangjh98/VCSR.git

# Run our processing script to generate test.json
python ../../HCCM/process_ERA_dataset.py VCSR/data/era_precomp test.json

# Return to the project root directory when done
cd ../../HCCM
```

### 2. Model Preparation

#### XVLM Pre-trained Model

Please download the pre-trained models for [X-VLM](https://github.com/zengyan-97/X-VLM) from the following links and save them to the `HCCM/pretrain` directory.

*   [X-VLM (4M, 200K steps)](https://drive.google.com/file/d/1B3gzyzuDN1DU0lvt2kDz2nTTwSKWqzV5/view?usp=sharing)
*   [X-VLM (16M, 200K steps)](https://drive.google.com/file/d/1iXgITaSbQ1oGPPvGaV0Hlae4QiJG5gx0/view?usp=sharing)

Once downloaded, execute the provided script to convert the original XVLM model weights into an MMCV-compatible format.

```bash
python process_xvlm2mmcv.py \
--input_path pretrain/16m_base_model_state_step_199999.th \
--output_path "pretrain/16m_base_model_state_step_199999_(xvlm2mmcv).pth"
```

#### BERT Model

Use the `huggingface-cli` tool to download the `bert-base-uncased` model to the `HCCM/pretrain` directory.

```bash
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir pretrain/bert-base-uncased
```

## Training HCCM on the GeoText1652 Dataset

By default, the training script automatically loads data from the `../datasets` path. If you have stored the datasets in a different location, please be sure to modify the `data_root` parameter in the following configuration files:

*   `HCCM/configs/_base_/datasets/geotext1652_retrieval.py`
*   `HCCM/configs/_base_/datasets/ERA_retrieval.py`

All of our experiments were conducted on a single 80G A800 GPU. By default, validation is performed once per epoch. Please note that we have not fully validated the code for multi-GPU training.

**Single-GPU Training:**

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/exp/xvlm_1xb24_hccm_geotext1652.py
```

**Special Instructions for Multi-GPU Training and Validation:**

Due to the large size of the full test set, the validation process has high VRAM requirements. To enable validation or testing on devices with limited resources (e.g., 24GB VRAM), we have implemented a strategy to offload data to the CPU. This means that **if periodic validation is enabled during training (i.e., `val_interval <= max_epochs`), the training can only be run on a single GPU**.

If you wish to use multiple GPUs for training, you **must disable periodic validation**. This can be done by setting the `val_interval` to a value greater than `max_epochs`. For example, modify the `train_cfg` in `configs/exp/xvlm_1xb24_hccm_geotext1652.py` as follows:

```python
# Set val_interval to a value greater than max_epochs to disable validation during multi-GPU training
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=10)
```

Multi-GPU training command (e.g., for 4 GPUs):

```bash
bash tools/dist_train.sh configs/exp/xvlm_1xb24_hccm_geotext1652.py 4
```

## Model Testing

You can download the HCCM model weights (`epoch_6.pth`), trained for 6 epochs on the GeoText1652 dataset, from [**here**](https://drive.google.com/file/d/1p468glkjTqxuE7YhzdXnC1Kx4xEJQEM3/view?usp=sharing).

### Testing on GeoText1652 Dataset

To perform a comprehensive bidirectional retrieval test (Image-to-Text and Text-to-Image), you need to perform the following two steps separately:

1.  **Test Image-to-Text (I2T) Direction:**
    *   Open the `configs/exp/xvlm_1xb24_hccm_geotext1652.py` file.
    *   Find the `test_cfg` parameter and set `i2t=True`.
    *   Execute the test command:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/exp/xvlm_1xb24_hccm_geotext1652.py \
    epoch_6.pth 
    ```

2.  **Test Text-to-Image (T2I) Direction:**
    *   Modify the `configs/exp/xvlm_1xb24_hccm_geotext1652.py` file again.
    *   Set the `i2t` parameter in `test_cfg` to `False`.
    *   Re-run the test command:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/exp/xvlm_1xb24_hccm_geotext1652.py \
    epoch_6.pth 
    ```

### Zero-Shot Testing on ERA Dataset

To evaluate the zero-shot generalization capability of HCCM on the ERA dataset, please run the following command. Similarly, you will need to modify the configuration file to test the I2T and T2I directions separately.

*   Open the `configs/exp/xvlm_1xb24_hccm_ERAzeroshot.py` file.
*   Depending on your testing needs, set the `i2t` parameter in `test_cfg` to `True` (for I2T) or `False` (for T2I).
*   Execute the test command:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
configs/exp/xvlm_1xb24_hccm_ERAzeroshot.py \
epoch_6.pth 
```

## Acknowledgements

Our research is built upon many excellent open-source projects. We would like to express our sincere gratitude to the developers and contributors of the following projects:

- [X-VLM](https://github.com/zengyan-97/X-VLM): A powerful pre-trained model for multi-granularity vision-language alignment that provided a solid foundation for our work.
- [GeoText-1652](https://github.com/MultimodalGeo/GeoText-1652): A high-quality benchmark dataset focused on spatial relationship matching for drone navigation.
- [OpenMMLab](https://github.com/open-mmlab): Thanks to [MMPretrain](https://github.com/open-mmlab/mmpretrain), [MMCV](https://github.com/open-mmlab/mmcv), and [MMEngine](https://github.com/open-mmlab/mmengine) for providing an efficient and scalable deep learning framework that greatly simplified our development process.