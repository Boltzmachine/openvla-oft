#!/bin/zsh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=240G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=2-00:00:00
#SBATCH --job-name=openvla-oft
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
#   --disentangle True \
#   --vla_path openvla/openvla-7b \
#   --data_root_dir dataset/ \
#   --dataset_name libero_spatial_no_noops \
#   --run_root_dir outputs/ \
#   --use_l1_regression True \
#   --use_diffusion False \
#   --use_film False \
#   --num_images_in_input 1 \
#   --use_proprio True \
#   --batch_size 8 \
#   --learning_rate 5e-4 \
#   --num_steps_before_decay 100000 \
#   --max_steps 150005 \
#   --save_freq 10000 \
#   --save_latest_checkpoint_only False \
#   --image_aug True \
#   --lora_rank 32 \
#   --wandb_entity neuroking \
#   --wandb_project openvla-oft \
#   --run_id_note openvla-oft-reproduce-nowrist

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir dataset/ \
  --dataset_name oxe_magic_soup_plus_minus \
  --run_root_dir outputs/ \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size 32 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 210005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity neuroking \
  --wandb_project openvla-oft \
  --run_id_note debug


  
