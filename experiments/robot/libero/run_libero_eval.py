"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

from copy import deepcopy
import json
import logging
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from libero.libero.envs.predicates import eval_predicate_fn

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from copy import deepcopy

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"
    LIBERO_MEMORY = "libero_memory"
    LIBERO_STOVE = "libero_stove"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
    TaskSuite.LIBERO_MEMORY: 512,  # longest training demo has 512 steps
    TaskSuite.LIBERO_STOVE: 600,  # longest training demo has 600 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    baseline: str = "none"                           # Baseline type (if applicable)
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action

def get_bounded_from_index(data, index):
    if index <= 0:
        if abs(index) > len(data):
            index = 0
    else:
        raise NotImplementedError("Positive indexing not supported.")
    return data[index]


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    episode_idx: int = 0,
):
    """Run a single episode in the environment."""
    # Reset environment
    if "memory" in cfg.task_suite_name or "stove" in cfg.task_suite_name:
        retry = 0
        while True:
            try:
                np.random.seed(episode_idx * 100 + retry)
                env.reset()
                env.env.init_moving_params()
                break
            except Exception as e:
                print(retry,e, file=open("log.txt", "a"))
                retry += 1
        env.env.moving_controller.counter = -10
        for _ in range(10):
            obs, reward, done, info = env.step(get_libero_dummy_action("llava"))
        assert env.env.moving_controller.counter == 0, "Environment failed to settle after reset!"
    else:
        env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.env._get_observations()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    replay_observations = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    max_cache_steps = 2 if model.config.static_ratio > 0 else 0
    cache_steps = 0

    # Run episode
    cache = None
    success = False
    replay_observations = []
    
    def get_past_observations(replay_observations, indices):
        past_obs = []
        for idx in indices:
            assert idx < 0
            if idx < 0:
                if -idx > len(replay_observations):
                    past_obs.append(replay_observations[0]['full_image'])
                else:
                    past_obs.append(replay_observations[idx]['full_image'])
        return past_obs
    
    def need_wait(task_name, env, t, num_steps_wait):
        if "memory" in task_name:
            return not env.env.moving_completed
        else:
            return t < num_steps_wait
    
    if "stove" in cfg.task_suite_name:
        cook_count = 0
        ever_moved = False
        
    while t < max_steps:
        # Do nothing for the first few timesteps to let objects stabilize
        if need_wait(cfg.task_suite_name, env, t, cfg.num_steps_wait):
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)
            replay_observations.append(deepcopy(observation))
            t += 1
            continue
        # Prepare observation
        observation, img = prepare_observation(obs, resize_size)
        replay_images.append(img)
        replay_observations.append(deepcopy(observation))
        observation = { 'full_image': get_past_observations(replay_observations, list(range(-1 - (model.vision_backbone.get_num_images_in_input()-1) * model.config.mem_sep, 0, model.config.mem_sep)) ) }
        # observation = { 'full_image': get_past_observations(replay_observations, [-100000, -1]) }
        # If action queue is empty, requery model
        if len(action_queue) == 0:
            history_index = -1 if cache is None else -9
            # observation = {'full_image': get_bounded_from_index(replay_observations, history_index)['full_image']}
            history_image = get_bounded_from_index(replay_observations, history_index)['full_image']
            # Query model to get action
            actions, cache = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
                history_image=None, #history_image,
                cache=None#cache,
            )
            action_queue.extend(actions)
            cache_steps += 1
            if cache_steps >= max_cache_steps:
                cache_steps = 0
                cache = None

        # Get action from queue
        action = action_queue.popleft()

        # Process action
        action = process_action(action, cfg.model_family)
        
        # Execute action in environment
        obs, reward, done, info = env.step(action.tolist())

        if "stove" in cfg.task_suite_name:
            stove = env.env.object_states_dict['flat_stove_1_cook_region']
            object_state = env.env.object_states_dict[env.env.first_cook_object]
            if eval_predicate_fn("on", object_state, stove):
                cook_count += 1
                
            if not eval_predicate_fn("closexy", object_state):
                ever_moved = True

        if done:
            success = True
            break
        t += 1

    # except Exception as e:
    #     log_message(f"Episode error: {e}", log_file)

    if 'stove' in cfg.task_suite_name:
        breakdown = {
            "cook_count": min(abs(cook_count - 28), 40)
        }
        goal_state = env.env.parsed_problem["goal_state"]
        for state in goal_state:
            if state[0] == 'closexy':
                breakdown[state[0]] = env.env._eval_predicate(state) and ever_moved
                success = success and breakdown[state[0]]
            else:
                breakdown[state[0]] = env.env._eval_predicate(state)

        return success, replay_images, t, breakdown
    return success, replay_images, t


def maybe_print_breakdown(cfg, breakdown, total_episodes, log_file, prestr):
    """Print breakdown results if applicable."""
    if "stove" in cfg.task_suite_name:
        for key in breakdown:
            if key == "cook_count":
                avg = np.mean([v for v in breakdown[key] if v > 0]) if len([v for v in breakdown[key] if v > 0]) > 0 else 0
            else:
                avg = np.mean(breakdown[key]) if len(breakdown[key]) > 0 else 0
            log_message(f"Overall {prestr} breakdown {key}: {avg}", log_file)


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    total_breakdown=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    try:
        initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    except:
        assert "memory" in cfg.task_suite_name or "stove" in cfg.task_suite_name, "Only libero_memory and libero_stove tasks can fail to load initial states!"
        initial_states, all_initial_states = None, None

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    if "stove" in cfg.task_suite_name:
        task_breakdown = defaultdict(list)
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)
        if initial_states is None:
            initial_state = None
        else:
            # Handle initial state
            if cfg.initial_states_path == "DEFAULT":
                # Use default initial state
                initial_state = initial_states[episode_idx]
            else:
                # Get keys for fetching initial episode state from JSON
                initial_states_task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{episode_idx}"

                # Skip episode if expert demonstration failed to complete the task
                if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                    log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                    continue

                # Get initial state
                initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, total_steps, *info = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            episode_idx
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        if len(info) > 0:
            breakdown = info[0]
            for key in breakdown:
                task_breakdown[key].append(breakdown[key])
                total_breakdown[key].append(breakdown[key])

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"Total steps: {total_steps}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        if "stove" in cfg.task_suite_name:
            for key in breakdown:
                log_message(f"{key}: {breakdown[key]}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)
    maybe_print_breakdown(cfg, task_breakdown, task_episodes, log_file, "current")
    maybe_print_breakdown(cfg, total_breakdown, total_episodes, log_file, "total")
    if "stove" in cfg.task_suite_name:
        return total_episodes, total_successes, total_breakdown

    # # Log to wandb if enabled
    # if cfg.use_wandb:
    #     wandb.log(
    #         {
    #             f"success_rate/{task_description}": task_success_rate,
    #             f"num_episodes/{task_description}": task_episodes,
    #         }
    #     )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    results_file = open("rollout.txt", "a")

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes, total_breakdown = 0, 0, defaultdict(list)
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes, *total_breakdown = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            total_breakdown,
            log_file,
        )
        if len(total_breakdown) > 0:
            total_breakdown = total_breakdown[0]
    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    maybe_print_breakdown(cfg, total_breakdown, total_episodes, log_file, "final")
    
    results_file.write(f"{cfg.pretrained_checkpoint}\t{final_success_rate:.4f}\n")
    results_file.close()
    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
