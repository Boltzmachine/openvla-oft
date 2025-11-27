torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --disentangle none \
  --static_ratio 0.8 \
  --mem_sep 40 \
  --vla_path openvla/openvla-7b \
  --data_root_dir dataset/ \
  --dataset_name libero_memory \
  --run_root_dir outputs \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 3 \
  --use_proprio False \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 170005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity neuroking \
  --wandb_project openvla-oft \
  --run_id_note memorydisent


# torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
#   --disentangle none \
#   --with_memory "[10000000, 1]"\
#   --vla_path openvla/openvla-7b \
#   --data_root_dir dataset/ \
#   --dataset_name libero_memory \
#   --run_root_dir outputs \
#   --use_l1_regression False \
#   --use_diffusion False \
#   --use_film False \
#   --num_images_in_input 1 \
#   --use_proprio False \
#   --batch_size 4 \
#   --grad_accumulation_steps 4 \
#   --learning_rate 5e-4 \
#   --num_steps_before_decay 100000 \
#   --max_steps 170005 \
#   --save_freq 10000 \
#   --save_latest_checkpoint_only False \
#   --image_aug True \
#   --lora_rank 32 \
#   --wandb_entity neuroking \
#   --wandb_project openvla-oft \
#   --run_id_note memory
