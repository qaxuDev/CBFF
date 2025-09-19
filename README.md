## Introduction

Welcome to the official repository of our paper "[*Cross Branch Feature Fusion Decoder for Consistency Regularization-based Semi-Supervised Change Detection*]([https://ieeexplore.ieee.org/document/10965597](https://ieeexplore.ieee.org/abstract/document/10446862))"! Our paper has been accepted by IEEE ICASSP 2024.

### Framework Overview

<div align="center">
<img src="figs/network.png" alt="CD network" width="500">
<p><em>The architecture of our change detection network. Features from two images are compared, fused via CBFFBlocks, and classified by dual heads with consistency regularization.</em></p>
</div>

<div align="center">
<img src="figs/SSCDframework.png" alt="SSCD Framework" width="500">
<p><em>The framework of consistency regularization-based semi-supervised change detection method.</em></p>
</div>

## Results
### 📊 Quantitative Results on WHU-CD and LEVIR-CD

The two numbers in each cell denote the **changed-class IoU** and **overall accuracy (OA)**, respectively.
Results reported under semi-supervised settings with varying labeled data ratios (5%, 10%, 20%, 40%). The highest scores are marked in \textbf{bold}.
#### 🔹 LEVIR-CD

| Method          | 5%        | 10%       | 20%       | 40%       |
|-----------------|-----------|-----------|-----------|-----------|
| AdvEnt          | 67.1 / 98.15 | 70.8 / 98.38 | 74.3 / 98.59 | 75.9 / 98.67 |
| s4GAN           | 66.6 / 98.16 | 72.2 / 98.48 | 75.1 / 98.63 | 76.2 / 98.68 |
| SemiCDNet       | 67.4 / 98.11 | 71.5 / 98.42 | 74.9 / 98.58 | 75.5 / 98.63 |
| SemiCD          | 74.2 / 98.59 | 77.1 / 98.74 | 77.9 / 98.79 | 79.0 / 98.84 |
| RC-CD           | 67.9 / 98.09 | 72.3 / 98.40 | 75.6 / 98.60 | 77.2 / 98.70 |
| SemiPTCD        | 71.2 / 98.39 | 75.9 / 98.65 | 76.6 / 98.65 | 77.2 / 98.74 |
| UniMatch        | 82.1 / 99.03 | 82.8 / 99.07 | 82.9 / 99.07 | 83.0 / 99.08 |
| **Ours**        | **82.6 / 99.05** | **83.2 / 99.08** | **83.2 / 99.09** | **83.9 / 99.12** |

#### 🔹 WHU-CD

| Method          | 5%        | 10%       | 20%       | 40%       |
|-----------------|-----------|-----------|-----------|-----------|
| AdvEnt          | 57.7 / 97.87 | 60.5 / 97.79 | 69.5 / 98.50 | 76.0 / 98.91 |
| s4GAN           | 57.3 / 97.94 | 58.0 / 97.81 | 67.0 / 98.41 | 74.3 / 98.85 |
| SemiCDNet       | 56.2 / 97.78 | 60.3 / 98.02 | 69.1 / 98.47 | 70.5 / 98.59 |
| SemiCD          | 65.8 / 98.37 | 68.0 / 98.45 | 74.6 / 98.83 | 78.0 / 99.01 |
| RC-CD           | 57.7 / 97.94 | 65.4 / 98.45 | 74.3 / 98.89 | 77.6 / 99.02 |
| SemiPTCD        | 74.1 / 98.85 | 74.2 / 98.86 | 76.9 / 98.95 | 80.8 / 99.17 |
| UniMatch        | 78.7 / 99.11 | 79.6 / 99.11 | 81.2 / 99.18 | 83.7 / 99.29 |
| **Ours**        | **81.0 / 99.20** | **81.1 / 99.18** | **83.6 / 99.29** | **86.5 / 99.43** |

✅ Our method achieves state-of-the-art performance across all settings, outperforming existing semi-supervised change detection methods on both WHU-CD and LEVIR-CD datasets. 

### Visualization Results
Below are qualitative comparisons on selected samples from WHU-CD and LEVIR-CD (5% labeled setting):

<div align="center">
<img src="figs/results.png" alt="Visualization Comparison: Input, GT, UniMatch, Ours" width="800">
<p><em> Detection results of different methods on WHU-CD and LEVIR-CD at the 5\% labeled training ratio. </em></p>
</div>


## Usage

### Installation

```bash
cd CBFF
conda create -n pytorch12 python=3.10.4
conda activate pytorch12
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Pretrained Backbone

[ResNet-50](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing).

```
├── ./pretrained
    ├── resnet50.pth
```



## Citation

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


## Acknowledgement

This project is based on [SemiCD](https://github.com/wgcban/SemiCD) and [UniMatch](https://github.com/LiheYoung/UniMatch). Thank you very much for their outstanding work.
