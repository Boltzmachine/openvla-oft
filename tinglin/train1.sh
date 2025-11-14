#!/bin/bash

#IMPORTANT: adjust batch_size and grad_accumulation_steps such that (batch_size * grad_accumulation_steps * num_gpus) = 64
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --seed 42 \
  --data_root_dir /mnt/hdfs/tinglin.huang/SEAgent/data/ola/modified_libero_rlds \
  --dataset_name libero_10_no_noops \
  --run_root_dir /mnt/hdfs/tinglin.huang/SEAgent/outputs/ola/ \
  --vla_path /mnt/hdfs/tinglin.huang/hg_model/qwen_custom/ \
  --use_l1_regression False \
  --num_images_in_input 1 \
  --use_proprio False \
  --use_diffusion False \
  --use_film False \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 180005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --disentangle none \
  --static_ratio 0.0 \
  --invswap_ratio 1.0 \
  --wandb_project ola \
  --wandb_entity test
