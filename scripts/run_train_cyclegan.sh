#!/usr/bin/env sh
set -ex
echo Begin running cycleGAN training script

python pytorch-CycleGAN-and-pix2pix train.py \
--dataroot ../result/AVN_NMDID_thickness_3_format_jpg_date0329_run0329 \
--name run0329 --model cycle_gan --save_epoch_freq 2 --batch_size 1 \
--checkpoints_dir ../checkpoints --lambda_mind 0