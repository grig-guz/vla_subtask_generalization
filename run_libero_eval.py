import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Union
from collections import  Counter
from torchvision.transforms import Resize
import torch
from PIL import Image
import hydra
import draccus
import numpy as np
from tqdm import tqdm
import logging
import cv2
import os
import imageio

from utils.calvin_utils import  get_libero_env, get_libero_dummy_action, load_octo_checkpoint, load_cogact_checkpoint, load_pi0_fast_checkpoint, invert_gripper_action
from utils.shared_utils import resize_image, high_to_low_level_mappings, get_action, normalize_gripper_action, set_seed_everywhere

from libero.libero import benchmark


logger = logging.getLogger(__name__)

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model: str = "octo"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    task_suite_name: str = "libero_spatial"
    num_sequences: int = 1000
    num_videos: int = 0
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # CALVIN environment-specific parameters
    #################################################################################################################
    video_save_dir: str = ''
    results_save_dir: str = ''

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under
    seed: int = 7                                    # Random Seed (for reproducibility)


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(video_save_dir, rollout_images, idx, success, task_description, timestep):
    """Saves an MP4 replay of an episode."""
    if video_save_dir != None:
        os.makedirs(video_save_dir, exist_ok=True)
        processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
        if timestep == -1:
            mp4_path = f"{video_save_dir}/episode={idx}--success={success}--task={processed_task_description}.mp4"
        else:
            mp4_path = f"{video_save_dir}/timestep={timestep}--episode={idx}--success={success}--task={processed_task_description}.mp4"
        video_writer = imageio.get_writer(mp4_path, fps=30)
        for img in rollout_images:
            video_writer.append_data(img)
        video_writer.close()
        print(f"Saved rollout MP4 at path {mp4_path}")
        return mp4_path

def evaluate_libero_policy(task_suite_name, num_trials_per_task, model_name, action_horizon, policy, num_steps_wait=10, timestep=-1, video_save_dir=None):

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {task_suite_name}")

    total_episodes, total_successes = 0, 0
    per_task_rates = {}

    for task_id in tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, lang_annotation = get_libero_env(task, resolution=256)

        if model_name == 'octo':
            goal = policy.create_tasks(texts=[lang_annotation])
            from octo.utils.train_callbacks import supply_rng
            policy_fn = supply_rng(
                    partial(
                        policy.sample_actions,
                        unnormalization_statistics=policy.dataset_statistics["action"],
                    ),
                )
            window_size = action_horizon
            act_step = action_horizon
        elif model_name == 'cogact':
            window_size = 10
            act_step = 10
        elif model_name == 'pi0_fast':
            window_size = 10
            act_step = 10


        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(range(num_trials_per_task)):
            print(f"\nTask: {lang_annotation}")
            if model_name == 'cogact':
                policy.reset()

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            max_steps = 400

            print(f"Starting episode {task_episodes+1}...")
            past_obs = None
            while t < max_steps + num_steps_wait:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action())
                    t += 1
                    past_obs = obs
                    continue
                
                resize_size = 224
                # Get preprocessed image
                img = get_libero_image(obs, resize_size)

                # Save preprocessed image for replay video
                replay_images.append(img)

                if model_name == 'octo':
                    if act_step > 0 and act_step % window_size == 0:
                        act_step = 0
                        
                        static_2 = resize_image(obs['agentview_image'][::-1], (256, 256), primary=True)
                        if past_obs:
                            static_1 = resize_image(past_obs['agentview_image'][::-1], (256, 256), primary=True)
                            image_primary = np.stack([static_1, static_2])
                            timestep_pad_mask = np.array([[True, True]])
                        else:
                            image_primary = np.stack([np.zeros((256, 256, 3)), static_2])
                            timestep_pad_mask = np.array([[False, True]])

                        pad_mask_dict = {
                            "image_primary": np.array([[True, True]]),
                            "timestep": np.array([[False, False]]),
                        }
                        
                        observation = {
                                "image_primary": np.expand_dims(image_primary, 0),  # uint8
                                "timestep_pad_mask": timestep_pad_mask,
                                "pad_mask_dict": pad_mask_dict,
                                "timestep": np.array([[t-1, t]]),
                        }
                        act_buffer = policy_fn(observation, goal)
                        act_buffer = np.array(act_buffer[0])
                        action = act_buffer[act_step]
                    else:
                        action = act_buffer[act_step]
                elif model_name == 'cogact':

                    if act_step > 0 and act_step % window_size == 0:
                        act_step = 0
                        image = resize_image(obs['agentview_image'][::-1], (224, 224), primary=True)
                        action_buffer = policy.step(image=image, task_description=lang_annotation)
                        action = np.array(action_buffer[act_step])
                    else:
                        action = np.array(action_buffer[act_step])
                elif model_name == 'pi0_fast':
                    if act_step > 0 and act_step % window_size == 0:
                        from openpi_client import image_tools
                        img = np.ascontiguousarray(obs["agentview_image"][::-1])
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, 224, 224)
                        )
                        act_step = 0
                        state = np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                        inputs = {"observation/image": img, "observation/state": state, "prompt": lang_annotation}
                        act_buffer = policy.infer(inputs)["actions"]
                        action = act_buffer[act_step]
                    else:
                        action = act_buffer[act_step]
                else:
                    raise Exception("Unknown model!")
                
                act_step += 1
                if model_name in ['octo', 'cogact']:
                    action = normalize_gripper_action(action)
                    action = invert_gripper_action(action)

                # Execute action in environment
                past_obs = obs
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                video_save_dir, replay_images, total_episodes, success=done, task_description=lang_annotation, timestep=timestep
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        
        #if cfg.use_wandb:
        #    wandb.log(
        #        {
        #            f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
        #            f"num_episodes/{task_description}": task_episodes,
        #        }
        #    )

        per_task_rates[lang_annotation] = float(task_successes) / float(task_episodes)
        # Log final results
        print(f"Current task success rate: {per_task_rates[lang_annotation]}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")


    total_rate = float(total_successes) / float(total_episodes)

    return total_rate, per_task_rates

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    # Set random seed
    set_seed_everywhere(cfg.seed)
    
    calvin_cfg = None
    if cfg.model == "octo":
        model, processor, calvin_cfg = load_octo_checkpoint(cfg.pretrained_checkpoint, cfg, calvin_cfg)
        cfg.action_horizon = 10
    elif cfg.model == 'cogact':
        model, _ = load_cogact_checkpoint(cfg.pretrained_checkpoint)
        cfg.action_horizon = 10
    elif cfg.model == 'pi0_fast':
        model, _ = load_pi0_fast_checkpoint(cfg.pretrained_checkpoint)
        cfg.action_horizon = 10


    average_rate, per_task_rate = evaluate_libero_policy(
        task_suite_name=cfg.task_suite_name,
        num_trials_per_task=cfg.num_trials_per_task,
        model_name=cfg.model,
        action_horizon=cfg.action_horizon,
        policy=model, 
        timestep=-1,
        num_steps_wait=10,
        video_save_dir=cfg.video_save_dir
    )

    results_dict = {}
    results_dict[Path(cfg.pretrained_checkpoint).name] = (average_rate, per_task_rate)
    pretr_path = Path(cfg.pretrained_checkpoint)

    if cfg.model == 'cogact':
        scores_path = Path(cfg.results_save_dir) / "libero" / (pretr_path.parent.parent.name + f"_{cfg.model}_results_{cfg.task_suite_name}.pkl")
    elif cfg.model == 'pi0_fast':
        scores_path = Path(cfg.results_save_dir) / "libero" / (pretr_path.parent.parent.name + "_" + pretr_path.parent.name + f"_{cfg.model}_results_{cfg.task_suite_name}.pkl")
    else:
        scores_path = Path(cfg.results_save_dir) / "libero" / (pretr_path.parent.name + "_" + pretr_path.name + f"_{cfg.model}_results_{cfg.task_suite_name}.pkl")

    with open(scores_path, 'wb') as myfile:
        pickle.dump(results_dict, myfile)

if __name__ == "__main__":
    eval_libero()
