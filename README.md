## Introduction

Welcome to the official repository of our paper "[*Cross Branch Feature Fusion Decoder for Consistency Regularization-based Semi-Supervised Change Detection*]([https://ieeexplore.ieee.org/document/10965597](https://ieeexplore.ieee.org/abstract/document/10446862))"! Our paper has been accepted by IEEE ICASSP 2024.

###Framework Overview

<div align="center">
<img src="figs/framework.png" alt="SSCD Framework Diagram" width="800">
<p><em>Figure: SSCD framework. Features from two images are compared, fused via CBFFBlocks, and classified by dual heads with consistency regularization.</em></p>
</div>

## Results
###Quantitative Comparison on WHU-CD and LEVIR-CD
The highest scores are marked in bold.
Results reported under semi-supervised settings with varying labeled data ratios (5%, 10%, 20%, 40%).

| Method | \multicolumn{8}{c|}{WHU-CD} | \multicolumn{8}{c}{LEVIR-CD} |
|--------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| | IoU | OA | IoU | OA | IoU | OA | IoU | OA | IoU | OA | IoU | OA | IoU | OA | IoU | OA |
| | 5% | 10% | 20% | 40% | 5% | 10% | 20% | 40% | 5% | 10% | 20% | 40% | 5% | 10% | 20% | 40% |
| AdvEnt~\cite{vu2019advent} | 57.7 | 60.5 | 69.5 | 76.0 | 97.87 | 97.79 | 98.50 | 98.91 | 67.1 | 70.8 | 74.3 | 75.9 | 98.15 | 98.38 | 98.59 | 98.67 |
| s4GAN~\cite{mittal2019semi} | 57.3 | 58.0 | 67.0 | 74.3 | 97.94 | 97.81 | 98.41 | 98.85 | 66.6 | 72.2 | 75.1 | 76.2 | 98.16 | 98.48 | 98.63 | 98.68 |
| SemiCDNet~\cite{peng2020semicdnet} | 56.2 | 60.3 | 69.1 | 70.5 | 97.78 | 98.02 | 98.47 | 98.59 | 67.4 | 71.5 | 74.9 | 75.5 | 98.11 | 98.42 | 98.58 | 98.63 |
| SemiCD~\cite{bandara2022revisiting} | 65.8 | 68.0 | 74.6 | 78.0 | 98.37 | 98.45 | 98.83 | 99.01 | 74.2 | 77.1 | 77.9 | 79.0 | 98.59 | 98.74 | 98.79 | 98.84 |
| RC-CD~\cite{wang2022reliable} | 57.7 | 65.4 | 74.3 | 77.6 | 97.94 | 98.45 | 98.89 | 99.02 | 67.9 | 72.3 | 75.6 | 77.2 | 98.09 | 98.40 | 98.60 | 98.70 |
| SemiPTCD~\cite{mao2023semi} | 74.1 | 74.2 | 76.9 | 80.8 | 98.85 | 98.86 | 98.95 | 99.17 | 71.2 | 75.9 | 76.6 | 77.2 | 98.39 | 98.65 | 98.65 | 98.74 |
| UniMatch~\cite{yang2023revisiting} | 78.7 | 79.6 | 81.2 | 83.7 | 99.11 | 99.11 | 99.18 | 99.29 | 82.1 | 82.8 | 82.9 | 83.0 | 99.03 | 99.07 | 99.07 | 99.08 |
| Ours (Proposed) | 81.0 | 81.1 | 83.6 | 86.5 | 99.20 | 99.18 | 99.29 | 99.43 | 82.6 | 83.2 | 83.2 | 83.9 | 99.05 | 99.08 | 99.09 | 99.12 |

✅ Our method achieves state-of-the-art performance across all settings, outperforming existing semi-supervised change detection methods on both WHU-CD and LEVIR-CD datasets. 

### Visualization Results
Below are qualitative comparisons on selected samples from WHU-CD and LEVIR-CD (5% labeled setting):

<div align="center">
<img src="figs/results.png" alt="Visualization Comparison: Input, GT, UniMatch, Ours" width="1200">
<p><em>Figure: Visual comparison of change detection results. </em></p>
</div>


## Usage





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
