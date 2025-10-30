from collections import Counter
from functools import reduce
import logging
from operator import add
import os
import sys
sys.path.append(".")

from typing import Any, Dict, Optional

import sys
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.distributed as dist
from torch.nn import Linear
import time



"""This script will collect data snt store it with a fixed window size"""

logger = logging.getLogger(__name__)



high_to_low_level_mappings = {
    # Basic block manipulation
    #"lift_red_block_table": [
    #    ["grasp_red_block", "lift_grasped_block"]
    #],
    #"lift_blue_block_table": [
    #    ["grasp_blue_block", "lift_grasped_block"]
    #],
    #"lift_pink_block_table": [
    #    ["grasp_pink_block", "lift_grasped_block"]
    #],

    # Lifting from slider
    #"lift_red_block_slider": [
    #    ["grasp_red_block", "lift_grasped_block"]
    #],
    #"lift_blue_block_slider": [
    #    ["grasp_blue_block", "lift_grasped_block"]
    #],
    #"lift_pink_block_slider": [
    #    ["grasp_pink_block", "lift_grasped_block"]
    #],

    # Lifting from drawer
    #"lift_red_block_drawer": [
    #    ["grasp_red_block", "lift_grasped_block"]
    #],
    #"lift_blue_block_drawer": [
    #    ["grasp_blue_block", "lift_grasped_block"]
    #],
    #"lift_pink_block_drawer": [
    #    ["grasp_pink_block", "lift_grasped_block"]
    #],

    # Pushing blocks
    "push_red_block_right": [
        ["contact_red_block_left", "push_block_right"]
    ],
    "push_red_block_left": [
        ["contact_red_block_right", "push_block_left"]
    ],
    "push_blue_block_right": [
        ["contact_blue_block_left", "push_block_right"]
    ],
    "push_blue_block_left": [
        ["contact_blue_block_right", "push_block_left"]
    ],
    "push_pink_block_right": [
        ["contact_pink_block_left", "push_block_right"]
    ],
    "push_pink_block_left": [
        ["contact_pink_block_right", "push_block_left"]
    ],

    # Slider operations
    "move_slider_left": [
        ["grasp_slider", "move_slider_left", "ungrasp_slider"]
    ],
    "move_slider_right": [
        ["grasp_slider", "move_slider_right", "ungrasp_slider"]
    ],

    # Drawer operations
    "open_drawer": [
        ["grasp_drawer", "open_drawer", "ungrasp_drawer"]
    ],
    "close_drawer": [
        ["grasp_drawer", "close_drawer", "ungrasp_drawer"]
    ],

    # Placing operations
    "place_red_block_in_slider": [
        ["grasp_red_block", "lift_grasped_block", "place_grasped_block_over_slider", "ungrasp_block"],
    ],
    "place_blue_block_in_slider": [
        ["grasp_blue_block", "lift_grasped_block", "place_grasped_block_over_slider", "ungrasp_block"],
    ],
    "place_pink_block_in_slider": [
        ["grasp_pink_block", "lift_grasped_block", "place_grasped_block_over_slider", "ungrasp_block"],
    ],
    "place_red_block_in_drawer": [
        ["grasp_red_block", "lift_grasped_block", "place_grasped_block_over_drawer", "ungrasp_block"],
    ],
    "place_blue_block_in_drawer": [
        ["grasp_blue_block", "lift_grasped_block", "place_grasped_block_over_drawer", "ungrasp_block"],
    ],
    "place_pink_block_in_drawer": [
        ["grasp_pink_block", "lift_grasped_block", "place_grasped_block_over_drawer", "ungrasp_block"],
    ],

    # Rotation operations (require grasping first)
    "rotate_red_block_right": [
        #["grasp_red_block", "lift_grasped_block", "rotate_grasped_block_right", "ungrasp_block"],
        ["grasp_red_block", "rotate_grasped_block_right", "ungrasp_block"],
    ],
    "rotate_red_block_left": [
        #["grasp_red_block", "lift_grasped_block", "rotate_grasped_block_left", "ungrasp_block"],
        ["grasp_red_block", "rotate_grasped_block_left", "ungrasp_block"]
    ],
    "rotate_blue_block_right": [
        #["grasp_blue_block", "lift_grasped_block", "rotate_grasped_block_right", "ungrasp_block"],
        ["grasp_blue_block", "rotate_grasped_block_right", "ungrasp_block"]
    ],
    "rotate_blue_block_left": [
        #["grasp_blue_block", "lift_grasped_block", "rotate_grasped_block_left", "ungrasp_block"],
        ["grasp_blue_block", "rotate_grasped_block_left", "ungrasp_block"]
    ],
    "rotate_pink_block_right": [
        #["grasp_pink_block", "lift_grasped_block", "rotate_grasped_block_right", "ungrasp_block"],
        ["grasp_pink_block", "rotate_grasped_block_right", "ungrasp_block"]
    ],
    "rotate_pink_block_left": [
        #["grasp_pink_block", "lift_grasped_block", "rotate_grasped_block_left", "ungrasp_block"],
        ["grasp_pink_block", "rotate_grasped_block_left", "ungrasp_block"]
    ],

    # Stack/unstack
    "stack_block": [
        # Red on Blue
        ["grasp_red_block", "lift_grasped_block", 
         "place_grasped_block_over_blue_block", "ungrasp_block"],
        # Red on Pink
        ["grasp_red_block", "lift_grasped_block", 
         "place_grasped_block_over_pink_block", "ungrasp_block"],
        # Blue on Red
        ["grasp_blue_block", "lift_grasped_block", 
         "place_grasped_block_over_red_block", "ungrasp_block"],
        # Blue on Pink
        ["grasp_blue_block", "lift_grasped_block", 
         "place_grasped_block_over_pink_block", "ungrasp_block"],
        # Pink on Red
        ["grasp_pink_block", "lift_grasped_block", 
         "place_grasped_block_over_red_block", "ungrasp_block"],
        # Pink on Blue
        ["grasp_pink_block", "lift_grasped_block", 
         "place_grasped_block_over_blue_block", "ungrasp_block"],
    ],
    "unstack_block": [
        # Red from blue
        #["grasp_red_block", "lift_grasped_block", 
        # "place_grasped_block_over_table", "ungrasp_block"],
        #["grasp_blue_block", "lift_grasped_block", 
        # "place_grasped_block_over_table", "ungrasp_block"],
        #["grasp_pink_block", "lift_grasped_block", 
        # "place_grasped_block_over_table", "ungrasp_block"],
        ["grasp_red_block", 
         "place_grasped_block_over_table", "ungrasp_block"],
        ["grasp_blue_block",  
         "place_grasped_block_over_table", "ungrasp_block"],
        ["grasp_pink_block",  
         "place_grasped_block_over_table", "ungrasp_block"],
    ]
}


def merge_data(list_of_data):

    merged_data = {
        "language": {"ann": [], "task": [], "emb": [], "subtasks_selected": [], "subtasks_actual": [], "satisfied": []},
        "info": {"episodes": [], "indx": []},
    }
    for d in list_of_data:
        for k in d:
            for k2, v2 in d[k].items():
                if isinstance(v2, list):
                    merged_data[k][k2] += v2
                elif isinstance(v2, np.ndarray) and len(merged_data[k][k2]) == 0:
                    merged_data[k][k2] = v2
                elif isinstance(v2, np.ndarray) and len(merged_data[k][k2]) != 0:
                    merged_data[k][k2] = np.concatenate((merged_data[k][k2], v2), axis=0)
                else:
                    print(type(v2))
                    raise ValueError
    return merged_data


class Annotator(Callback):
    def __init__(self, cfg):
        self.envs = None  # type: Any
        self.cfg = cfg
        self.device = None
        self.lang_folder = cfg.lang_folder
        self.tasks = hydra.utils.instantiate(cfg.callbacks.rollout_lh.tasks)
        self.low_level_tasks = hydra.utils.instantiate(cfg.callbacks.rollout_lh.low_level_tasks)
        self.demo_task_counter_train = Counter()  # type: Counter[str]
        self.demo_task_counter_val = Counter()  # type: Counter[str]
        self.train_dataset = None
        self.val_dataset = None
        self.file_name = "auto_lang_ann.npy"  # + save_format
        self.train_lang_folder = None
        self.val_lang_folder = None
        self.collected_data_train = {
            "language": {"ann": [], "task": [], "emb": [], "subtasks_selected": [], "subtasks_actual": [], "satisfied": []},
            "info": {"episodes": [], "indx": []},
        }  # type: Dict
        self.collected_data_val = {
            "language": {"ann": [], "task": [], "emb": [], "subtasks_selected": [], "subtasks_actual": [], "satisfied": []},
            "info": {"episodes": [], "indx": []},
        }  # type: Dict
        self.lang_model = None
        self.num_samples_train = None
        self.num_samples_val = None
        self.finished_annotation_val = False
        self.scene_idx_info = None

    @rank_zero_only
    def create_folders(self):
        self.train_lang_folder = self.train_dataset.abs_datasets_dir / self.lang_folder
        self.train_lang_folder.mkdir(parents=True, exist_ok=True)

        self.val_lang_folder = self.val_dataset.abs_datasets_dir / self.lang_folder
        self.val_lang_folder.mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def compute_val_embeddings(self):
        val_sent = self.cfg.val_instructions
        embeddings = {}
        for task, ann in val_sent.items():
            embeddings[task] = {}
            language_embedding = self.lang_model(list(ann))
            embeddings[task]["emb"] = language_embedding.cpu().numpy()
            embeddings[task]["ann"] = ann
        np.save(self.val_lang_folder / "embeddings", embeddings)
        logger.info("Done saving val language embeddings for Rollouts !")

    def init_vars(self, trainer, pl_module):
        self.device = pl_module.device
        self.val_dataset = trainer.val_dataloaders[0].dataset.datasets["vis"]  # type: ignore
        self.train_dataset = trainer.train_dataloader.dataset.datasets["vis"]
        self.scene_idx_info = np.load(self.train_dataset.abs_datasets_dir / "scene_info.npy", allow_pickle=True).item()

        self.envs = {
            scene: hydra.utils.instantiate(
                self.cfg.callbacks.rollout_lh.env_cfg, self.val_dataset, pl_module.device, scene=scene
            )
            for scene, _ in self.scene_idx_info.items()
        }
        
        if self.cfg.validation_scene not in self.envs:
            self.envs[self.cfg.validation_scene] = hydra.utils.instantiate(
                self.cfg.callbacks.rollout_lh.env_cfg,
                self.val_dataset,
                pl_module.device,
                scene=self.cfg.validation_scene,
                cameras=(),
            )

        self.create_folders()
        self.lang_model = hydra.utils.instantiate(self.cfg.model)
        self.compute_val_embeddings()
        self.num_samples_train = int(self.cfg.eps * len(self.train_dataset) / len(self.cfg.train_instructions.keys()))
        self.num_samples_val = int(self.cfg.eps * len(self.val_dataset) / len(self.cfg.train_instructions.keys()))

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the validation loop begins."""
        if self.envs is None:
            self.init_vars(trainer, pl_module)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.envs is None:
            self.init_vars(trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        batch = batch["vis"] if isinstance(batch, dict) else batch
        self.collected_data_val, self.demo_task_counter_val, current_task_counter = self.annotate(
            batch,
            self.val_dataset,
            self.collected_data_val,
            self.demo_task_counter_val,
            self.num_samples_val,
            trainer.current_epoch
        )
        if dist.is_available() and dist.is_initialized():
            global_counters = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(global_counters, current_task_counter)
            current_task_counter = reduce(add, global_counters)
        self.demo_task_counter_val += current_task_counter
        if self.check_done(
            self.demo_task_counter_val, self.num_samples_val, batch_idx, trainer.num_val_batches[0], "val"
        ):
            print()
            print()
            print()
            logger.info("Finished annotating val dataset")
            print()
            print()
            print()
            self.finished_annotation_val = True
            self.save_and_postprocess(self.collected_data_val, self.val_lang_folder, "val", len(self.val_dataset))

    def on_train_batch_end(  # type: ignore
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        batch = batch["vis"] if isinstance(batch, dict) else batch

        self.collected_data_train, self.demo_task_counter_train, current_task_counter = self.annotate(
            batch, self.train_dataset, self.collected_data_train, self.demo_task_counter_train, self.num_samples_train, trainer.current_epoch
        )
        if dist.is_available() and dist.is_initialized():
            global_counters = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(global_counters, current_task_counter)
            current_task_counter = reduce(add, global_counters)
        self.demo_task_counter_train += current_task_counter
        if self.check_done(
            self.demo_task_counter_train, self.num_samples_train, batch_idx, trainer.num_training_batches, "train", trainer.current_epoch
        ):
            print()
            print()
            print()
            print()
            print()
            print()
            pl_module.finished_annotation_train = True  # type: ignore
            self.save_and_postprocess(self.collected_data_train, self.train_lang_folder, "train", len(self.train_dataset))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, unused: Optional[int] = None) -> None:
        self.save_and_postprocess(self.collected_data_train, self.train_lang_folder, "train", len(self.train_dataset))
        self.train_dataset.min_window_size = self.train_dataset.min_window_size // 2
        self.train_dataset.max_window_size = self.train_dataset.max_window_size // 2
        self.train_dataset.episode_lookup = self.train_dataset.build_file_indices(self.train_dataset.abs_datasets_dir)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_and_postprocess(self.collected_data_val, self.val_lang_folder, "val", len(self.val_dataset))
        self.val_dataset.min_window_size = self.val_dataset.min_window_size // 2
        self.val_dataset.max_window_size = self.val_dataset.max_window_size // 2
        self.val_dataset.episode_lookup = self.val_dataset.build_file_indices(self.val_dataset.abs_datasets_dir)


    def save_and_postprocess(self, collected_data, lang_folder, mod, length):
        if dist.is_available() and dist.is_initialized():
            global_collected_data = [None for _ in range(dist.get_world_size())]
            torch.distributed.all_gather_object(global_collected_data, collected_data)
            if dist.get_rank() == 0:
                global_collected_data = merge_data(global_collected_data)
                np.save("lang_ann", global_collected_data)
        else:
            np.save("lang_ann", collected_data)
            
        if self.cfg.postprocessing:
            language = collected_data["language"]["ann"]
            language_embedding = self.lang_model(language)
            collected_data["language"]["emb"] = language_embedding.cpu().numpy()
            logger.info(f"Done extracting {mod} language embeddings !")

        if dist.is_available() and dist.is_initialized():
            global_collected_data = [None for _ in range(dist.get_world_size())]
            torch.distributed.all_gather_object(global_collected_data, collected_data)
            if dist.get_rank() != 0:
                return
            collected_data = merge_data(global_collected_data)
            print()
        print("Tasks not collected: ", [key for key in self.cfg.train_instructions.keys() if key not in self.demo_task_counter_train.keys()])
        np.save(self.file_name, collected_data)
        np.save(lang_folder / self.file_name, collected_data)
        logger.info(f"Done saving {mod} language annotations !")

        lang_length = float(len(collected_data["language"]["ann"]))
        logger.info(
            f"\nVision Dataset contains  {length} datapoints "
            f"\nLanguage Dataset contains {lang_length} datapoints "
            f"\n    VISION --> {100.0 * length / (length + lang_length):.3f} %"
            f"\n    LANGUAGE --> {100.0 * lang_length / (length + lang_length):.3f} %"
        )

    def check_done(self, counter, num_samples, batch_idx, num_batches, mode, epoch=None):
        if batch_idx % 10 == 0:
            print(f"{mode} Epoch {epoch} Tasks Objective: {num_samples}")
            print(f"Tasks Lang: {self.cfg.train_instructions.keys()}")
            print(f"Done batches: {batch_idx} Total batches: {num_batches}")
            print(f"Tasks Annotations Progress: {counter}")
            print(f"Tasks Not Detected: {set(self.cfg.train_instructions.keys()) - set(counter.keys())}")
            print(
                "Progress [ "
                + "=" * int(0.5 * 100 * batch_idx / num_batches)
                + ">"
                + "-" * int(0.5 * 100 * (num_batches - batch_idx) / num_batches)
                + str(round(100 * batch_idx / num_batches))
                + "%"
                + "]"
            )
        return len(counter.values()) >= len(self.cfg.train_instructions) and min(counter.values()) >= num_samples

    def select_env(self, dataset, idx):
        if "validation" in dataset.abs_datasets_dir.as_posix():
            return self.envs[self.cfg.validation_scene]
        seq_idx = dataset.episode_lookup[idx]
        for scene, interval in self.scene_idx_info.items():
            if interval[0] <= seq_idx <= interval[1]:
                #print("SELECTING ENV, SEQ INDEX IS ", seq_idx)
                return self.envs[scene]
        raise ValueError

    def annotate(self, episode, dataset, collected_data, global_task_counter, num_samples, current_epoch):
        # Each episode is 32 batches, length 64 each.
        state_obs = episode["robot_obs"]
        reset_info = episode["state_info"]
        idx = episode["idx"]

        batch_size, seq_length = state_obs.shape[0], state_obs.shape[1]
        current_task_counter = Counter()

        quarter = seq_length // 4
        half = seq_length // 2
        start = time.time()
        for i in range(batch_size):
            env = self.select_env(dataset, idx[i])
            # reset env to state of last step in the episode (goal state)
            env.reset(reset_info, i, -1)
            goal_info = env.get_info()

            prior_steps = np.random.randint(quarter, half)
            env.reset(reset_info, i, prior_steps)
            # State somewhere between a quarter and a half of the window (16-32 for 64-length window) 
            middle_info = env.get_info()
            
            env.reset(reset_info, i, seq_length - quarter)
            # State at the last quarter of the episode (48 for 64-length window)
            close_to_end_info = env.get_info()

            task_info = self.tasks.get_task_info(middle_info, goal_info)

            if (
                len(task_info) != 1 # 0 or more than 1 instructions completed
                or not task_info <= self.cfg.train_instructions.keys() # not applicable to our case
                or len(self.tasks.get_task_info_for_set(middle_info, close_to_end_info, task_info)) # the given task is completed before step 48 
            ):
                continue

            task = list(task_info)[0]

            # Don't collect too much of the same task
            # TODO: Below would make data not IID. Fix?
            if global_task_counter[task] + current_task_counter[task] >= num_samples:
                continue

            # reset self.env to state of first step in the episode
            env.reset(reset_info, i, 0)
            start_info = env.get_info()

            env.reset(reset_info, i, half)
            middle_info2 = env.get_info()


            if len(self.tasks.get_task_info_for_set(start_info, goal_info, task_info)) and not len(
                self.tasks.get_task_info(start_info, middle_info2)
            ):
                # If the task is completed after step 32,
                # start index is same as in the idx specified above
                start_idx = idx[i]
                window_size = seq_length
                prior_steps = 0
            else:
                # otherwise, (if there is some (potentially other task) completed between 0 and 32)
                # shorten the task window by prior steps on both sides?
                # I guess this is for tasks which are "short"
                start_idx = idx[i] + prior_steps
                window_size = seq_length - prior_steps

            low_level_seq_satisfied, selected_subtask_seq, no_duplicates_suffix, subtasks_suffix = self.check_low_level_task_seq(
                env=env,
                reset_info=reset_info,
                start_timestep=prior_steps,
                end_timestep=prior_steps + window_size, 
                batch_idx=i, 
                task=task
            )
            if (not no_duplicates_suffix) or len(subtasks_suffix) > 0:
                #print(f"Found subtasks in traj suffix! {subtasks_suffix}, no_duplicates: {no_duplicates_suffix}")
                continue

            no_duplicates, actual_subtask_seq = self.determine_low_level_task_seq(
                env=env,
                reset_info=reset_info,
                start_timestep=prior_steps,
                end_timestep=prior_steps + window_size, 
                batch_idx=i
            )
            

            if not no_duplicates:
                #print("Found duplicates!")
                continue
            
            # seq_length = torch.unique(actions[i], dim=0).shape[0]
            collected_data, added = self.label_seq(
                collected_data=collected_data, 
                dataset=dataset, 
                seq_length=window_size, 
                idx=start_idx, 
                task=task, 
                low_level_seq_satisfied=low_level_seq_satisfied, 
                selected_subtask_seq=selected_subtask_seq, 
                actual_subtask_seq=actual_subtask_seq
            )
            
            if added:
                current_task_counter += Counter(task_info)

            """
            if dataset.episode_lookup[start_idx] == 505654:
                _, actual_subtask_seq_2 = self.determine_low_level_task_seq(
                    env=env,
                    reset_info=reset_info,
                    start_timestep=prior_steps,
                    seq_length=prior_steps+window_size, 
                    batch_idx=i,
                    log_infos=True
                )

                breakpoint()
            """
            #num_collected = len(collected_data["info"]["indx"])
            #print(f"Current epoch: {current_epoch}, Collected: {num_collected} \nTask: {task}, Resulting length: {window_size} selected: {selected_subtask_seq}, actual: {actual_subtask_seq} \n")
        print("Processed batches: ", time.time() - start)
        return collected_data, global_task_counter, current_task_counter



    def check_low_level_task_seq(self, env, reset_info, start_timestep, end_timestep, batch_idx, task):
        # Need to check that only a particular sequence of low-level tasks gets completed.
        # Iterate over all   
        low_level_task_sequences = high_to_low_level_mappings[task]
        for low_level_task_seq in low_level_task_sequences:
            subtask_idx = 0
            subtask = low_level_task_seq[subtask_idx]
            env.reset(reset_info, batch_idx, start_timestep)
            start_info = env.get_info()
            end_info = {} 
            completed_sequence = False
            
            for i in range(start_timestep + 1, end_timestep):
                env.reset(reset_info, batch_idx, i)
                end_info = env.get_info()

                done, terminate = self.low_level_tasks.get_task_info_with_criteria(start_info, end_info, subtask)
                if terminate:
                    break
                elif not done:
                    continue
                else:
                    # Found all subtasks
                    if subtask_idx == len(low_level_task_seq) - 1:
                        no_duplicates, other_tasks_completed = self.determine_low_level_task_seq(
                            env=env,
                            reset_info=reset_info,
                            start_timestep=i,
                            end_timestep=end_timestep,
                            batch_idx=batch_idx
                        )

                        return True, low_level_task_seq, no_duplicates, other_tasks_completed
                    else:
                        subtask_idx += 1
                        try:
                            subtask = low_level_task_seq[subtask_idx]
                        except IndexError:
                            breakpoint()

                    env.reset(reset_info, batch_idx, i)
                    start_info = env.get_info()

        return completed_sequence, [], True, []
        
        


    def determine_low_level_task_seq(self, env, reset_info, start_timestep, end_timestep, batch_idx, log_infos=False):
        # Need to check that only a particular sequence of low-level tasks gets completed.
        # Iterate over all   
        if log_infos:
            print(reset_info["scene_obs"][batch_idx, start_timestep])
        env.reset(reset_info, batch_idx, start_timestep)
        start_info = env.get_info()
        subtasks_completed = []
        
        for i in range(start_timestep + 1, end_timestep):
            env.reset(reset_info, batch_idx, i)
            if log_infos:
                print(reset_info["scene_obs"][batch_idx, i])

            end_info = env.get_info()

            tasks_completed = self.low_level_tasks.get_task_info(start_info, end_info)
            if len(tasks_completed) == 0:
                continue
            elif len(tasks_completed) > 1:
                print(f"Found duplicate subtasks!!! {tasks_completed}")
                return False, []
            else:
                subtasks_completed.append(list(tasks_completed)[0])
                start_info = end_info

        return True, subtasks_completed
    


    def label_seq(self, collected_data, dataset, seq_length, idx, task, low_level_seq_satisfied, selected_subtask_seq, actual_subtask_seq):
        
        try:
            seq_idx = dataset.episode_lookup[idx]
        except Exception:
            print("Got out of bounds! ", idx, len(dataset.episode_lookup))
            return collected_data, False


        if self.check_intersections(collected_data, seq_idx, seq_length, task):
            return collected_data, False
        
        collected_data["info"]["indx"].append((seq_idx, seq_idx + seq_length))
        task_lang = self.cfg.train_instructions[task]
        lang_ann = task_lang[np.random.randint(len(task_lang))]
        collected_data["language"]["ann"].append(lang_ann)
        collected_data["language"]["task"].append(task)
        collected_data["language"]["subtasks_selected"].append(selected_subtask_seq)
        collected_data["language"]["subtasks_actual"].append(actual_subtask_seq)
        collected_data["language"]["satisfied"].append(low_level_seq_satisfied)


        return collected_data, True

    def check_intersections(self, collected_data, seq_idx, seq_length, task):
        return False
        for i, (stored_seq_start, stored_seq_end) in enumerate(collected_data["info"]["indx"]):
            if ((stored_seq_start <= seq_idx <= stored_seq_end) or (stored_seq_start <= seq_idx + seq_length <= stored_seq_end) or (seq_idx <= stored_seq_start <= stored_seq_end <= seq_idx + seq_length) or (stored_seq_start <= seq_idx <= seq_idx + seq_length <= stored_seq_end)) and task == collected_data["language"]["task"][i]:
                return True
        return False

class LangAnnotationModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.finished_annotation_train = False
        self.dummy_net = Linear(1, 1)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:  # type: ignore
        if self.finished_annotation_train:
            return -1

    def training_step(self, batch, batch_idx):
        return self.dummy_net(torch.Tensor([0.0]).to(self.device))

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


@hydra.main(config_path="../../conf", config_name="lang_ann.yaml")
def main(cfg: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(cfg.seed)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    callbacks = Annotator(cfg)
    dummy_model = LangAnnotationModel()
    print("SET LIMIT TRAIN BATCHES")

    trainer_args = {
        **cfg.trainer,
        "callbacks": callbacks,
        "num_sanity_val_steps": 0,
        "max_epochs": 3,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        #"limit_train_batches": 20,
        #"limit_val_batches": 20,
    }

    print("Devices:", trainer_args["devices"])
    # Configure multi-GPU training
    if trainer_args["devices"] > 1:  # type: ignore
        trainer_args["strategy"] = DDPStrategy(find_unused_parameters=False)
    print(trainer_args)

    trainer = Trainer(**trainer_args)

    trainer.fit(dummy_model, datamodule=datamodule)
    #trainer.validate(dummy_model, datamodule=datamodule)  # type: ignore


if __name__ == "__main__":
    main()
