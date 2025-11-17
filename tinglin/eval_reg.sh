#!/bin/bash

cp /mnt/bn/uv-wlv2/dev /opt/tiger -rf

cd /opt/tiger/dev

source /mnt/bn/uv-wlv2/envs/oft/bin/activate

python experiments/robot/libero/run_libero_eval.py \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --pretrained_checkpoint <CKPT> \
  --task_suite_name libero_10

mv rollout.txt <TARGET_PATH>