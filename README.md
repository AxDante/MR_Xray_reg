# MR_Xray_reg

**(This repository is still under development.)**

**MSR_MOGA** is a project that mainly aims to perform annotation registration between pre-operative MR images and intra-operative X-rays for core decompressing surgery. The packages included in this repository are as follow:
 * ImagePreprocessing
 * ImageSimilarity
 * pytorch-CycleGAN-and-pix2pix (Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
 * CTtoXray (In progress)

## Prerequisites
* Python >= 3.7.0 
* torch >= 1.5.0
* CPU or NVIDIA GPU + CUDA CuDNN

## Model Introduction
 (In progress)

## How to Use (Linux environment)
### Installation
- Run the following pip command to install required packages
```
pip install -r ./pytorch-CycleGAN-and-pix2pix/requirements.txt
```
### Run
- Dataset preprocessing script
```
chmod +x ./scripts/run_preprocessing.sh
./scripts/run_preprocessing.sh
```
- CycleGAN training script
```
chmod +x ./scripts/run_cyclegan.sh
./scripts/run_cyclegan.sh
```
- CycleGAN validation script (In progress)
```
chmod +x ./scripts/run_validation.sh
./scripts/run_validation.sh
```
## Contact
Feel free to contact me through pku1@jhu.edu if you have any questions.
