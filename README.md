# 🌍 Cross Branch Feature Fusion Decoder for Consistency Regularization-based Semi-Supervised Change Detection

Welcome to the official repository of our paper "[*Cross Branch Feature Fusion Decoder for Consistency Regularization-based Semi-Supervised Change Detection*](https://ieeexplore.ieee.org/abstract/document/10446862)"! Our paper has been accepted by IEEE ICASSP 2024.

Our method introduces **CBFF (Cross Branch Feature Fusion)** — a novel decoder that synergistically combines the **local expressiveness of CNNs** and the **global reasoning of Transformers**, enabling high-performance change detection even under extreme label scarcity. Through comprehensive experiments on WHU-CD and LEVIR-CD, we demonstrate consistent superiority over seven state-of-the-art semi-supervised methods.

---

## 🔍 Introduction

Semi-supervised change detection (SSCD) aims to detect pixel-level changes between bi-temporal remote sensing images using a small amount of labeled data and abundant unlabeled imagery — a critical setting given the high cost of manual annotation.

While recent works explore Transformer-based architectures for global context modeling, we observe (as shown in Fig. 1 of our paper) that **convolutional decoders significantly outperform transformer-only decoders under low-label regimes** (e.g., 5% labeled data). This is because Transformers require large volumes of annotated samples to converge stably.

To bridge this gap, we propose **CBFF**: a hierarchical decoder that fuses two complementary branches:
- **Local Convolutional Branch (LCB)**: Learns robust local features efficiently with minimal supervision.
- **Global Transformer Branch (GTB)**: Captures long-range contextual dependencies across image pairs.

By combining these branches via residual fusion and enforcing **strong-to-weak consistency regularization** on unlabeled data, CBFF achieves superior performance without requiring massive labeled datasets.

---

## 🏗️ Framework

Our network follows a structured pipeline:
- **Difference Feature Generator**: Uses a Siamese ResNet backbone to extract features from two input images; computes change features via absolute difference and two convolution layers. 
- **Bottleneck (ASPP)**: Applies Atrous Spatial Pyramid Pooling on the deepest feature to capture multi-scale context. 
- **CBFF Modules (×3)**: Hierarchical fusion blocks integrating LCB and GTB at each resolution level, enabling stable learning with few labels.
- **Dual Prediction Heads**: Two parallel classifiers generate change maps from fused features, enabling consistency loss during training.

<div align="center">
  <img src="figs/network.png" alt="CBFF Decoder Architecture" width="480">
  <img src="figs/SSCDframework.png" alt="Training Framework: Strong-to-Weak Consistency" width="480">
</div>

<p style="text-align: center; color: #666; font-size: 0.9em; margin-top: 10px;">
  <em><strong>Left:</strong> CBFF decoder fuses CNN and Transformer features at each level.<br>
  <strong>Right:</strong> Training uses supervised CE loss on labeled data and unsupervised consistency loss on unlabeled data.</em>
</p>

---

## 📊 Results

### 1. Quantitative Results on WHU-CD and LEVIR-CD

The two numbers in each cell denote the **changed-class IoU** and **overall accuracy (OA)**, respectively.  
Results are reported under semi-supervised settings with varying labeled ratios: **5%, 10%, 20%, 40%**.  
**Bold values indicate state-of-the-art performance**.

#### 🔹 WHU-CD

| Method       | 5%         | 10%        | 20%        | 40%        |
|--------------|------------|------------|------------|------------|
| AdvEnt       | 57.7 / 97.87 | 60.5 / 97.79 | 69.5 / 98.50 | 76.0 / 98.91 |
| s4GAN        | 57.3 / 97.94 | 58.0 / 97.81 | 67.0 / 98.41 | 74.3 / 98.85 |
| SemiCDNet    | 56.2 / 97.78 | 60.3 / 98.02 | 69.1 / 98.47 | 70.5 / 98.59 |
| SemiCD       | 65.8 / 98.37 | 68.0 / 98.45 | 74.6 / 98.83 | 78.0 / 99.01 |
| RC-CD        | 57.7 / 97.94 | 65.4 / 98.45 | 74.3 / 98.89 | 77.6 / 99.02 |
| SemiPTCD     | 74.1 / 98.85 | 74.2 / 98.86 | 76.9 / 98.95 | 80.8 / 99.17 |
| UniMatch     | 78.7 / 99.11 | 79.6 / 99.11 | 81.2 / 99.18 | 83.7 / 99.29 |
| **Ours**     | **81.0 / 99.20** | **81.1 / 99.18** | **83.6 / 99.29** | **86.5 / 99.43** |

#### 🔹 LEVIR-CD

| Method       | 5%         | 10%        | 20%        | 40%        |
|--------------|------------|------------|------------|------------|
| AdvEnt       | 67.1 / 98.15 | 70.8 / 98.38 | 74.3 / 98.59 | 75.9 / 98.67 |
| s4GAN        | 66.6 / 98.16 | 72.2 / 98.48 | 75.1 / 98.63 | 76.2 / 98.68 |
| SemiCDNet    | 67.4 / 98.11 | 71.5 / 98.42 | 74.9 / 98.58 | 75.5 / 98.63 |
| SemiCD       | 74.2 / 98.59 | 77.1 / 98.74 | 77.9 / 98.79 | 79.0 / 98.84 |
| RC-CD        | 67.9 / 98.09 | 72.3 / 98.40 | 75.6 / 98.60 | 77.2 / 98.70 |
| SemiPTCD     | 71.2 / 98.39 | 75.9 / 98.65 | 76.6 / 98.65 | 77.2 / 98.74 |
| UniMatch     | 82.1 / 99.03 | 82.8 / 99.07 | 82.9 / 99.07 | 83.0 / 99.08 |
| **Ours**     | **82.6 / 99.05** | **83.2 / 99.08** | **83.2 / 99.09** | **83.9 / 99.12** |

> ✅ Our method consistently outperforms all baselines, achieving **+1.5%~+2.8% IoU gain on WHU-CD** and **+0.3%~+0.9% IoU gain on LEVIR-CD** over UniMatch.

---

### 2. Visualization Results

Qualitative comparison on selected samples from **WHU-CD** and **LEVIR-CD** (5% labeled training ratio):

<div align="left">
  <img src="figs/results.png" alt="Visualization Comparison: Input, GT, UniMatch, Ours" width="800">
  <p><em>Detection results of different methods under 5% labeled ratios.</em></p>
</div>

---

## 🚀 Getting Started

### Installation

```bash
cd CBFF
conda create -n pytorch12 python=3.10.4
conda activate pytorch12
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Pretrained Backbone

Download the pre-trained ResNet-50 checkpoint: [ResNet-50](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing).

```
├── ./pretrained
    ├── resnet50.pth
```

### Dataset

- WHU-CD: [imageA, imageB, and label](https://www.dropbox.com/s/r76a00jcxp5d3hl/WHU-CD-256.zip?dl=0)
- LEVIR-CD: [imageA, imageB, and label](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip?dl=0)

Please modify your dataset path in configuration files.

```
├── [Your WHU-CD/LEVIR-CD Path]
    ├── A
    ├── B
    └── label
```
> ✅ Update the dataset path in your config file (e.g., config.yaml) before training. 

---

## 🛠️ Usage

### ▶️ Training & Testing

```bash
cd CBFF
bash scripts/train.sh
bash scripts/test.sh
```

Modify parameters in ``scripts/train.sh``:

- dataset: Set to ``whu`` or ``levir``.
- split: Set to ``5%``, ``10%``, ``20%``, ``40%``.
- method: Use ``supervised`` for supervised training, ``fixmatch_CbffDecoder`` for consistency training.

---

## 📝 Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{xing2024cross,
  title={Cross branch feature fusion decoder for consistency regularization-based semi-supervised change detection},
  author={Xing, Yan and Xu, Qi’ao and Zeng, Jingcheng and Huang, Rui and Gao, Sihua and Xu, Weifeng and Zhang, Yuxiang and Fan, Wei},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={9341--9345},
  year={2024},
  organization={IEEE}
}
```


## ❤️ Acknowledgements

This project is based on [SemiCD](https://github.com/wgcban/SemiCD) and [UniMatch](https://github.com/LiheYoung/UniMatch). Thank you very much for their outstanding work.
