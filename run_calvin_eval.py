import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Union
from collections import  Counter
from torchvision.transforms import Resize
import torch
import hydra
import draccus
import numpy as np
from tqdm import tqdm
import logging
import cv2
import os

from utils.calvin_utils import get_env_state_for_initial_condition, get_env_and_checkpoint, resize_image, add_text
from utils.rollout_video import RolloutVideo

from utils.multistep_sequences import get_sequences
from utils.multistep_sequences_low_level_random import get_low_level_random_sequences
from utils.multistep_sequences_high2low_chain import get_sequences_high2low_chain
from utils.shared_utils import high_to_low_level_mappings, normalize_gripper_action, set_seed_everywhere

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model: str = "octo"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    task_suite_name: str = "libero_spatial"
    eval_type: str = None
    num_sequences: int = 1000
    num_videos: int = 0

    #################################################################################################################
    # CALVIN environment-specific parameters
    #################################################################################################################
    calvin_config_path: str = "/ubc/cs/research/nlp/grigorii/projects/low_level_tasks/conf/med_tasks_config.yaml"
    video_save_dir: str = ""
    results_save_dir:str = ''

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under
    seed: int = 7                                    # Random Seed (for reproducibility)
    # fmt: on

def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success

def get_video_tag(i):
    return f"_long_horizon/sequence_{i}"

def evaluate_policy(cfg, model, processors, env, task_oracle, annotations, eval_sequences, counters, seeds_dict=None, num_sequences=1000):
    task_oracle = hydra.utils.instantiate(task_oracle)

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
    if eval_sequences is None:
        if cfg.eval_type == 'train_low_random':
            eval_sequences = get_low_level_random_sequences(num_sequences)
        elif cfg.eval_type == 'train_high':
            eval_sequences = get_sequences(num_sequences)
        elif cfg.eval_type == 'train_conjunction':
            eval_sequences = get_sequences_high2low_chain(num_sequences)
        else:
            raise Exception("The eval sequences were not passed!@")
        
    eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for i, (initial_state, high_level_task, eval_seq) in enumerate(eval_sequences):

        record = i < cfg.num_videos
        if counters is not None:
            counters['high_level_started'][high_level_task] += 1

        result = evaluate_sequence(
            eval_sequence=eval_seq, 
            i=i,
            initial_state=initial_state, 
            high_level_task=high_level_task,
            cfg=cfg,
            model=model, 
            processors=processors,
            env=env, 
            task_oracle=task_oracle, 
            annotations=annotations,
            record=record,
            rollout_video=rollout_video,
            counters=counters,
            seeds_dict=seeds_dict
        )
        print("Eval seq: ", eval_seq)
        print("Result: ", result)
        results.append(result)
        if result == len(eval_seq) and counters is not None:
            counters['high_level_completed'][high_level_task] += 1
        elif cfg.eval_type == 'conjunction' and result == 1:
            counters['high_level_completed'][high_level_task] += 1


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
    eval_sequence, i, initial_state, high_level_task, cfg, model, processors, env, task_oracle, annotations, record, rollout_video, 
    counters, seeds_dict
):
    if counters is not None:
        counters["errors"].append([])

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state, seeds_dict)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)

    success_counter = 0
    print("Evaluating sequence: ", eval_sequence)

    for subtask in eval_sequence:
        if cfg.eval_type in ['train_conjunction', 'conj_random', 'conj_random_easy_eval']:
            high_task, low_subtasks = subtask
            lang_annotation = ", then ".join([annotations[low_subtask][0] for low_subtask in low_subtasks])
            subtask = high_task
        else:
            lang_annotation = annotations[subtask][0]
        print("Evaluating task ", subtask)
        if counters is not None:
            counters['low_level_started'][subtask] += 1

        if record:
            rollout_video.new_subtask()

        success = rollout(
            task=subtask,
            lang_annotation=lang_annotation,
            cfg=cfg,
            model=model,
            processors=processors,
            env=env, 
            task_oracle=task_oracle,
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


def rollout(task, lang_annotation, cfg, model, processors, env, task_oracle, record=False, rollout_video=None, counters=None):

    obs = env.get_obs()
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
        
    start_info = env.get_info()
    past_obs = None
    action_buffer = None
    
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
        obs, _, _, current_info = env.step(action)

        if record:
            # update video
            frame_aug = torch.zeros((3, 224, 448))
            resize = Resize(224, antialias=True)
            frame_aug[:, :, :224] = resize(torch.tensor(obs["rgb_obs"]["rgb_static"]).permute(2, 0, 1))
            closest_obs = 0
            if isinstance(closest_obs, int):
                closest_obs = torch.zeros((3, 224, 224))
            frame_aug[:, :, 224:] = closest_obs.squeeze()
            rollout_video.update(frame_aug.unsqueeze(0).unsqueeze(0), step=step)

        # check if current step solves a task
        if cfg.eval_type not in ['low_random_easy_eval', 'high_random_easy_eval', 'conj_random_easy_eval']:
            done, terminate = task_oracle.get_task_info_with_criteria(start_info, current_info, task)
            if terminate:
                wrong_tasks = task_oracle.get_task_info(start_info=start_info, end_info=current_info) - task_oracle.admissible_constraints[task]
                log_run_result(counters, task, lang_annotation, wrong_tasks, record, rollout_video)
                return False

            if done:
                log_run_result(counters, task, lang_annotation, "success", record, rollout_video)
                return True
        else:
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
            if len(current_task_info) > 0:
                log_run_result(counters, task, lang_annotation, "success", record, rollout_video)
                return True
            
    log_run_result(counters, task, lang_annotation, "timeout", record, rollout_video)

    return False

def get_action(cfg, model, processors, obs, past_obs, lang_annotation, goal, act_step, action_buffer, window_size, step):
    if act_step > 0 and act_step % window_size == 0:
        primary_img = obs['rgb_obs']['rgb_static']
        gripper_img = obs['rgb_obs']['rgb_gripper']
        state = obs['robot_obs']
        act_step = 0

        if cfg.model == 'octo':
                static_2 = resize_image(primary_img, (256, 256), primary=True)
                gripper_2 = resize_image(gripper_img, (128, 128))
                if past_obs:
                    static_1 = resize_image(past_obs['rgb_obs']['rgb_static'], (256, 256), primary=True)
                    gripper_1 = resize_image(past_obs['rgb_obs']['rgb_gripper'], (128, 128))
                    primary_img_stacked = np.stack([static_1, static_2])
                    image_wrist = np.stack([gripper_1, gripper_2])
                    timestep_pad_mask = np.array([[True, True]])
                else:
                    primary_img_stacked = np.stack([np.zeros((256, 256, 3)), static_2])
                    image_wrist = np.stack([np.zeros((128, 128, 3)), gripper_2])
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
def eval_calvin(cfg: GenerateConfig) -> None:
    # Set random seed
    set_seed_everywhere(cfg.seed)
    
    model, processor, env, calvin_cfg = get_env_and_checkpoint(cfg)
    calvin_cfg.num_sequences = cfg.num_sequences
    calvin_cfg.num_videos = cfg.num_videos
    calvin_cfg.eval_type = cfg.eval_type
    calvin_cfg.model = cfg.model
    calvin_cfg.video_save_dir = cfg.video_save_dir
    calvin_cfg.hard_eval = True

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

    high_level_tasks = OmegaConf.load(Path(__file__).parents[0] / "calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    low_level_tasks = OmegaConf.load(Path(__file__).parents[0] / "calvin/calvin_models/conf/callbacks/rollout/tasks/low_level_tasks.yaml")
    high_level_task_ann = OmegaConf.load(Path(__file__).parents[0] / "calvin/calvin_models/conf/annotations/new_playtable_validation.yaml")
    low_level_task_ann = OmegaConf.load(Path(__file__).parents[0] / "calvin/calvin_models/conf/annotations/low_level_tasks_validation.yaml")

    print(cfg.eval_type)
    
    if cfg.eval_type in ['high_random', 'high_random_easy_eval']:
        task_oracle = high_level_tasks
        task_annotation = high_level_task_ann
        eval_sequences = get_sequences(calvin_cfg.num_sequences)
        with open('utils/high_sequences_seeds', 'rb') as f:
            seeds_dict = pickle.load(f)
    elif cfg.eval_type in ['conj_random', 'conj_random_easy_eval']:
        task_oracle = high_level_tasks
        task_annotation = low_level_task_ann
        eval_sequences = get_sequences_high2low_chain(calvin_cfg.num_sequences)
        with open('utils/high2low_seeds', 'rb') as f:
            seeds_dict = pickle.load(f)
    elif cfg.eval_type in ['low_random', 'low_random_easy_eval']:
        task_oracle = low_level_tasks
        task_annotation = low_level_task_ann
        eval_sequences = get_low_level_random_sequences(calvin_cfg.num_sequences)
        with open('utils/low_sequence_seeds', 'rb') as f:
            seeds_dict = pickle.load(f)
    else:
        raise Exception("Unknown task type!")

    results, average_rate, success_rates, counters = evaluate_policy(
                    cfg=calvin_cfg,
                    model=model,
                    processors=processor,
                    env=env,
                    task_oracle=task_oracle, 
                    annotations=task_annotation, 
                    eval_sequences=eval_sequences, 
                    counters=counters,
                    seeds_dict=seeds_dict
    )
    
    results_dict = {}
    results_dict[Path(cfg.pretrained_checkpoint).name] = (average_rate, success_rates, counters)
    pretr_path = Path(cfg.pretrained_checkpoint)
    subset_stats_path = Path(cfg.pretrained_checkpoint) / 'subset_stats.pkl'
    if os.path.isfile(subset_stats_path):
        with open(subset_stats_path, "rb") as input_file:
            subset_stats = pickle.load(input_file)
            results_dict['subset_stats'] = subset_stats

    if cfg.model in ['cogact']:
        scores_path = Path(cfg.results_save_dir) / "calvin" / (pretr_path.parent.parent.name + f"_{cfg.model}_results_{cfg.eval_type}.pkl")
    elif cfg.model in ['pi05']:
        scores_path = Path(cfg.results_save_dir) / "calvin" / (pretr_path.parent.parent.parent.name + f"_{cfg.model}_results_{cfg.eval_type}.pkl")
    else:
        scores_path = Path(cfg.results_save_dir) / "calvin" / (pretr_path.parent.name + "_" + pretr_path.name + f"_{cfg.model}_results_{cfg.eval_type}.pkl")

    with open(scores_path, 'wb') as myfile:
        pickle.dump(results_dict, myfile)

if __name__ == "__main__":
    eval_calvin()
