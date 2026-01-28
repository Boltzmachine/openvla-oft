SPATIAL_CKPT=

for baseline in base ttf flashvla vlacache
do
python experiments/robot/libero/run_libero_eval.py \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --pretrained_checkpoint ${SPATIAL_CKPT} \
  --baseline ${baseline} \
  --task_suite_name libero_spatial
done


GOAL_CKPT=

for baseline in base ttf flashvla vlacache
do
python experiments/robot/libero/run_libero_eval.py \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --pretrained_checkpoint ${GOAL_CKPT} \
  --baseline ${baseline} \
  --task_suite_name libero_goal
done


OBJECT_CKPT=

for baseline in base ttf flashvla vlacache
do
python experiments/robot/libero/run_libero_eval.py \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --pretrained_checkpoint ${OBJECT_CKPT} \
  --baseline ${baseline} \
  --task_suite_name libero_object
done