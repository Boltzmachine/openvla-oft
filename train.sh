#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=720G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=2-00:00:00
#SBATCH --job-name=memory
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --seed 2025 \
  --disentangle none \
  --static_ratio 0.9 \
  --mem_sep 20 \
  --vla_path openvla/openvla-7b \
  --data_root_dir dataset/ \
  --dataset_name libero_stove \
  --run_root_dir outputs \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 20 \
  --use_proprio False \
  --batch_size 6 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 100005 \
  --save_freq 2000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity neuroking \
  --wandb_project openvla-oft \
  --run_id_note stove-20-20

#baseline
#torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
#  --seed 2025 \
#  --disentangle none \
#  --static_ratio 0.0 \
#  --mem_sep 20 \
#  --vla_path openvla/openvla-7b \
#  --data_root_dir dataset/ \
#  --dataset_name libero_stove \
#  --run_root_dir outputs \
#  --use_l1_regression False \
#  --use_diffusion False \
#  --use_film False \
#  --num_images_in_input 1 \
#  --use_proprio False \
#  --batch_size 8 \
#  --grad_accumulation_steps 4 \
#  --learning_rate 5e-4 \
#  --num_steps_before_decay 100000 \
#  --max_steps 100005 \
#  --save_freq 10000 \
#  --save_latest_checkpoint_only False \
#  --image_aug True \
#  --lora_rank 32 \
#  --wandb_entity neuroking \
#  --wandb_project openvla-oft \
#  --run_id_note stove-20-20-ttf
