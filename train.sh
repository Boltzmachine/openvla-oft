export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --seed 42 \
  --data_root_dir dataset/ \
  --dataset_name libero_10_no_noops \
  --run_root_dir outputs/ \
  --use_l1_regression True \
  --num_images_in_input 1 \
  --use_proprio False \
  --use_diffusion False \
  --use_film False \
  --batch_size 4 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 180005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity neuroking \
  --wandb_project openvla-oft