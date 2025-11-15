#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=3:00:00
#SBATCH --job-name=libero
#SBATCH --output=outputs/libero/slurms/%j.out
#SBATCH --qos=qos_nmi


python experiments/robot/libero/run_libero_eval.py \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --pretrained_checkpoint <CKPT> \
  --task_suite_name libero_spatial