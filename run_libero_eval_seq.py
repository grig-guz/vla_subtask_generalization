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
from collections import defaultdict

from utils.calvin_utils import  get_libero_env, load_octo_checkpoint, load_cogact_checkpoint, load_pi0_fast_checkpoint, invert_gripper_action, resize_image
from utils.shared_utils import  high_to_low_level_mappings, normalize_gripper_action, set_seed_everywhere, get_libero_dummy_action, quat2axisangle




logger = logging.getLogger(__name__)

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model: str = "octo"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    num_sequences: int = 1000
    num_videos: int = 0
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    eval_type: str = ""
    ep_len: int = 500
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

def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1,::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
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

def evaluate_libero_policy_seq(cfg, model, processors, eval_sequences, counters):
    from libero.libero.envs import OffScreenRenderEnv
    if counters is not None:
        counters["errors"] = []

    # video stuff
    if cfg.num_videos > 0:
        rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=cfg.video_save_dir,
            resolution_scale=1,
        )
    else:
        rollout_video = None

    results = []
    if cfg.num_sequences < len(eval_sequences):
        eval_sequences = eval_sequences[:cfg.num_sequences]
    

    env_args = {"bddl_file_name": "/home/gguz/projects/aip-vshwartz/gguz/vla_subtask_generalization/LIBERO/libero/libero/bddl_files/libero_single/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet.bddl", 
                    "camera_heights": 256, 
                    "camera_widths": 256}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)


    eval_sequences = tqdm(eval_sequences, position=0, leave=True)
    
    for i, (_, high_level_task, eval_seq, initial_state) in enumerate(eval_sequences):

        record = i < cfg.num_videos
        result = evaluate_sequence(
            eval_sequence=eval_seq, 
            i=i,
            initial_state=initial_state, 
            high_level_task=high_level_task,
            cfg=cfg,
            model=model, 
            processors=processors,
            env=env, 
            record=record,
            rollout_video=rollout_video,
            counters=counters,
        )
        print("Result: ", result)
        results.append(result)

        if record:
            rollout_video.write_to_tmp()

        success_rates = count_success(results)
        average_rate = sum(success_rates) / len(success_rates) * 5
        description = " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_rates)])
        description += f" Average: {average_rate:.1f} |"
        eval_sequences.set_description(description)

    if cfg.num_videos > 0:
        # log rollout videos
        rollout_video._log_videos_to_file(0, save_as_video=False)
        
    if counters is not None:
        for key, counter in counters.items():
            print(key, counter)

    return results, average_rate, success_rates, counters


def evaluate_sequence(
    eval_sequence, i, initial_state, high_level_task, cfg, model, processors, env, record, rollout_video, 
    counters
):
    if counters is not None:
        counters["errors"].append([])


    env.reset()
    env.set_init_state(initial_state)
    env.env.set_seq_evaluation(False)
    # Important to handle temporal predicates (some might reference object init states which are not initialized yet)
    env.env.parsed_problem["goal_state"] = [env.env.task_to_predicate[eval_sequence[0]]]

    t = 0
    if t < cfg.num_steps_wait:
        obs, reward, done, info = env.step(get_libero_dummy_action())
        t += 1
        past_obs = obs

    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)

    success_counter = 0
    print("Evaluating sequence: ", eval_sequence)

    observations = (obs, past_obs)

    for subtask in eval_sequence:
        env.env.parsed_problem["goal_state"] = [env.env.task_to_predicate[subtask]]
        
        if cfg.eval_type in ['libero_conj_single']:
            lang_annotation = ", then ".join([env.env.task_to_lang[low_subtask] for low_subtask in env.env.task_to_subtasks[subtask]])
            print(lang_annotation)
        else:
            lang_annotation = env.env.task_to_lang[subtask]

        print(f"Evaluating task {subtask} with lang {lang_annotation}")
        
        if counters is not None:
            counters['low_level_started'][subtask] += 1

        if record:
            rollout_video.new_subtask()

        success, observations = rollout(
            observations=observations,
            task=subtask,
            lang_annotation=lang_annotation,
            cfg=cfg,
            model=model,
            processors=processors,
            env=env, 
            record=record, 
            rollout_video=rollout_video,
            counters=counters
        )

        if record:
            rollout_video.draw_outcome(success)
        if success:
            if counters is not None:
                counters['low_level_completed'][subtask] += 1
            success_counter += 1
        else:
            return success_counter
        
    return success_counter


def rollout(observations, task, lang_annotation, cfg, model, processors, env, record=False, rollout_video=None, counters=None):
    
    goal = None
    if cfg.model == 'octo':
        from octo.utils.train_callbacks import supply_rng
        policy_fn = supply_rng(
                partial(
                    model.sample_actions,
                    unnormalization_statistics=model.dataset_statistics["action"],
                ),
            )
        goal = model.create_tasks(texts=[lang_annotation])
        model = policy_fn
    elif cfg.model == 'cogact':
        model.reset()

    window_size = cfg.action_horizon
    act_step = cfg.action_horizon

    if cfg.model in ['smolvla', 'groot', 'pi05']:
        window_size = 1
        act_step = 1
        model.config.n_action_steps = 10
        model.reset()
        
    action_buffer = None
    obs, past_obs = observations

    for step in range(cfg.ep_len):
        
        action, action_buffer, act_step = get_action(
            cfg=cfg,
            model=model,
            processors=processors,
            obs=obs,
            past_obs=past_obs,
            lang_annotation=lang_annotation,
            goal=goal,
            act_step=act_step,
            action_buffer=action_buffer,
            window_size=window_size,
            step=step
        )
        act_step += 1
        past_obs = obs
        obs, _, done, info = env.step(action)

        if record:
            # update video
            frame_aug = torch.zeros((3, 256, 512))
            resize = Resize(256, antialias=True)
            frame_aug[:, :, :256] = resize(past_obs['agentview_image'][::-1,::-1]).permute(2, 0, 1)
            closest_obs = 0
            if isinstance(closest_obs, int):
                closest_obs = torch.zeros((3, 256, 256))
            frame_aug[:, :, 256:] = closest_obs.squeeze()
            rollout_video.update(frame_aug.unsqueeze(0).unsqueeze(0), step=step)

        # check if current step solves a task
        if cfg.eval_type not in ['libero_low_level_single_easy'] and not info["hard_eval_passed"]:
            wrong_tasks = info["inadmissible_task"]
            log_run_result(counters, task, lang_annotation, wrong_tasks, record, rollout_video)
            return False, (obs, past_obs)

        if done:
            log_run_result(counters, task, lang_annotation, "success", record, rollout_video)
            return True, (obs, past_obs)
            
    log_run_result(counters, task, lang_annotation, "timeout", record, rollout_video)

    return False, (obs, past_obs)


def get_action(cfg, model, processors, obs, past_obs, lang_annotation, goal, act_step, action_buffer, window_size, step):

    if act_step > 0 and act_step % window_size == 0:
        primary_img = obs['agentview_image'][::-1, ::-1].copy()
        state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
        act_step = 0

        if cfg.model == 'octo':
                static_2 = resize_image(primary_img, (256, 256), primary=True)
                if past_obs:
                    static_1 = resize_image(past_obs['agentview_image'][::-1,::-1], (256, 256), primary=True)
                    primary_img_stacked = np.stack([static_1, static_2])
                    timestep_pad_mask = np.array([[True, True]])
                else:
                    primary_img_stacked = np.stack([np.zeros((256, 256, 3)), static_2])
                    timestep_pad_mask = np.array([[False, True]])

                pad_mask_dict = {
                    "image_primary": np.array([[True, True]]),
                    "timestep": np.array([[False, False]]),
                }
                
                observation = {
                        "image_primary": np.expand_dims(primary_img_stacked, 0),  # uint8
                        "timestep_pad_mask": timestep_pad_mask,
                        "pad_mask_dict": pad_mask_dict,
                        "timestep": np.array([[step-1, step]]),
                }
                #if 'wrist' in cfg.image_obs_keys:
                #    observation['image_wrist'] = np.expand_dims(image_wrist, 0)
                #    pad_mask_dict["image_wrist"] = np.array([[True, True]])
                action_buffer = model(observation, goal)
                action_buffer = np.array(action_buffer[0])

        elif cfg.model == 'cogact':
            action_buffer = model.step(image=primary_img, task_description=lang_annotation)
            action_buffer = np.array(action_buffer)
        elif cfg.model == 'pi0_fast':
            inputs = {"observation/image": primary_img, "observation/state": state, "prompt": lang_annotation}
            action_buffer = model.infer(inputs)["actions"]
            action_buffer = np.array(action_buffer)
        elif cfg.model in ['smolvla', 'groot', 'pi05']:
            preprocessor, postprocessor = processors
            main_img = torch.permute(torch.tensor(primary_img, device="cuda:0", dtype=torch.float32), (2, 0, 1)).unsqueeze(0)
            observation = {
                "observation.images.camera1": main_img.div(255),
                "observation.images.camera2": torch.zeros(((1, 3, 256, 256)), device="cuda:0"),
                #'observation.images.camera2_is_pad': torch.tensor([True], device="cuda:0"),
                "observation.state": torch.tensor(state, dtype=torch.float32, device="cuda:0"),
                "task": lang_annotation
            }

            proc_observation = preprocessor(observation)
            action_buffer = model.select_action(proc_observation)
            # LeRobot handles action chunks on its own
            action_buffer = np.array(postprocessor(action_buffer))
            action = action_buffer[0]
            return action, action_buffer, act_step
        else:
            raise Exception("Unknown model!")

    action = action_buffer[act_step]
    action = normalize_gripper_action(action)
    return action, action_buffer, act_step


def log_run_result(counters, task, lang_annotation, result, record, rollout_video):
    if record:
        rollout_video.add_language_instruction(lang_annotation)

    if counters != None:
        counters["errors"][-1].append((task, lang_annotation, result))


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
    elif cfg.model in ['pi05', 'smolvla', 'groot']:
        model, processor = load_smolvla_groot_checkpoint(checkpoint_path)
        cfg.action_horizon = 1

    with open('utils/libero_low_sequences_init_states', 'rb') as f:
        eval_sequences = pickle.load(f)
        
    high_level_started = Counter()
    high_level_completed = Counter()
    low_level_started = Counter()
    low_level_completed = Counter()

    counters = {
        'high_level_started': high_level_started,
        'high_level_completed': high_level_completed,
        'low_level_started': low_level_started,
        'low_level_completed': low_level_completed,
    }
    

    
    results, average_rate, success_rates, counters = evaluate_libero_policy_seq(
        cfg=cfg,
        model=model, 
        processors=processor,
        eval_sequences=eval_sequences,
        counters=counters,
    )

    results_dict = {}
    results_dict[Path(cfg.pretrained_checkpoint).name] = (average_rate, success_rates, counters)
    pretr_path = Path(cfg.pretrained_checkpoint)

    if cfg.model == 'cogact':
        scores_path = Path(cfg.results_save_dir) / "libero_new" / (pretr_path.parent.parent.name + f"_{cfg.model}_results_{cfg.eval_type}.pkl")
    elif cfg.model in ['pi05']:
        scores_path = Path(cfg.results_save_dir) / "libero_new" / (pretr_path.parent.parent.parent.name + f"_{cfg.model}_results_{cfg.eval_type}.pkl")
    else:
        scores_path = Path(cfg.results_save_dir) / "libero_new" / (pretr_path.parent.name + "_" + pretr_path.name + f"_{cfg.model}_results_{cfg.eval_type}.pkl")

    with open(scores_path, 'wb') as myfile:
        pickle.dump(results_dict, myfile)


if __name__ == "__main__":
    eval_libero()
