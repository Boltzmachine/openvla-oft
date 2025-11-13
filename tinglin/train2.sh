#!/bin/bash

#IMPORTANT: adjust batch_size and grad_accumulation_steps such that (batch_size * grad_accumulation_steps * num_gpus) = 64
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --seed 42 \
  --data_root_dir dataset/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir outputs/ \
  --use_l1_regression False \
  --num_images_in_input 1 \
  --use_proprio False \
  --use_diffusion False \
  --use_film False \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 210005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --disentangle none \
  --static_ratio 0.5 \
  --invswap_ratio 0.2 