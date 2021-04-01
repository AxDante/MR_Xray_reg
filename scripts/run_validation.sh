#!/usr/bin/env sh
set -ex
echo Begin running validation script
python pytorch-CycleGAN-and-pix2pix/test_full_volume.py \
--dataroot ../result/AVN_NMDID_thickness_3_format_jpg_date0329_run0329 \
--name run0329 --model cycle_gan --no_dropout \
--checkpoints_dir ../checkpoints --results_dir ../test_results/run0329


python pytorch-CycleGAN-and-pix2pix/test_full_volume.py --dataroot ../result/AVN_NMDID_thickness_3_format_jpg_date0329_run0329 --name run0329 --model cycle_gan --no_dropout --checkpoints_dir ../checkpoints --results_dir ../test_results/run0329
