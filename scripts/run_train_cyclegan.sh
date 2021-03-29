#!/usr/bin/env sh
set -ex
echo Begin running cycleGAN training script

python pytorch-CycleGAN-and-pix2pix train.py --dataroot ../datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout