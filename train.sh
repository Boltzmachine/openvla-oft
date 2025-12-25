#!/bin/zsh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=480G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=2-00:00:00
#SBATCH --job-name=openvla-oft
#SBATCH --output=outputs/slurms/%j.out

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --seed 42 \
  --vla_path outputs/libero_10_no_noops+b32+lr-0.0005+lora-r32+swap-0.2+dis-none+static-0.8+gate-True+bw-82--gate--40000_chkpt \
  --data_root_dir dataset/ \
  --dataset_name libero_10_no_noops \
  --run_root_dir outputs/ \
  --use_l1_regression True \
  --num_images_in_input 1 \
  --use_proprio False \
  --use_diffusion False \
  --use_film False \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 180005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --use_lora False \
  --lora_rank 32 \
  --wandb_entity neuroking \
  --wandb_project openvla-oft \
  --disentangle none \
  --use_contrastive True \
  --use_cache_gate True \
  --static_ratio 0.8 \
  --invswap_ratio 0.2 \
  --backward_window_size 82 \
  --run_id_note gate \
  --resume 1 \
  --resume_dir outputs/libero_10_no_noops+b32+lr-0.0005+lora-r32+swap-0.2+dis-none+static-0.8+gate-True+bw-82--gate--40000_chkpt/

# torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
#   --seed 42 \
#   --vla_path outputs/gate3-libero_10_no_noops+b32+lr-0.0005+lora-r32+swap-0.2+dis-none+static-0.8+bw-18--62000_chkpt/ \
#   --data_root_dir dataset/ \
#   --dataset_name libero_10_no_noops \
#   --run_root_dir outputs/ \
#   --use_l1_regression True \
#   --num_images_in_input 1 \
#   --use_proprio False \
#   --use_diffusion False \
#   --use_film False \
#   --batch_size 4 \
#   --grad_accumulation_steps 4 \
#   --learning_rate 5e-4 \
#   --num_steps_before_decay 100000 \
#   --max_steps 180005 \
#   --save_freq 10000 \
#   --save_latest_checkpoint_only False \
#   --image_aug True \
#   --use_lora False \
#   --wandb_entity neuroking \
#   --wandb_project openvla-oft \
#   --disentangle none \
#   --use_contrastive False \
#   --use_cache_gate True \
#   --static_ratio 0.8 \
#   --invswap_ratio 0.2 \
#   --backward_window_size 18 \
#   --train_gate_only True \
#   --resume 1 \
#   --resume_dir outputs/gate3-libero_10_no_noops+b32+lr-0.0005+lora-r32+swap-0.2+dis-none+static-0.8+bw-18--62000_chkpt/