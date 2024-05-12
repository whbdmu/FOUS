# Fast One-Stage Unsupervised Domain Adaptive Person Search

## Introduction
This is the official implementation for our paper Fast One-Stage Unsupervised Domain Adaptive Person Search (FOUS) in IJCAI2024.

Performance :
we tried some hyper-parameters and got better ReID performance reported in our paper.

|  Source   |  Target   | mAP  | Top-1 |                             CKPT                             |
| :-------: | :-------: | :--: | :---: | :----------------------------------------------------------: |
|    PRW    | CUHK-SYSU | 78.7 | 80.7  | [ckpt](https://drive.google.com/file/d/1pPAr284Onjl1FsyrVDKwbkhLKg5I-Inm/view?usp=drive_link) |
| CUHK-SYSU |    PRW    | 35.4 | 80.8  | [ckpt](https://drive.google.com/file/d/1c3WHC6ntSMAVl8Ys35ZTaseZFUII6Yj1/view?usp=drive_link) |


## Installation

Install Nvidia [Apex](https://github.com/NVIDIA/apex)

Run `pip install -r requirements.txt` in the root directory of the project.


## Data Preparation

1. Download [CUHK-SYSU](https://drive.google.com/open?id=1z3LsFrJTUeEX3-XjSEJMOBrslxD2T5af) and [PRW](https://goo.gl/2SNesA) datasets, and unzip them.
2. Modify `configs/prw_da.yaml` and `configs/cuhk_sysu_da.yaml` to change the dataset store place (Line 1,5,6) to your own path.
