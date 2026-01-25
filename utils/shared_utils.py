import os
import torch
import numpy as np
import random 


from pathlib import Path
from omegaconf import OmegaConf
import math
import contextlib


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def run_validation(policy, eval_type, model_name, global_step, processors=None, config_path="utils/med_tasks_config.yaml"):
    
    if 'single' in eval_type:
        from run_libero_eval_seq import evaluate_libero_policy
        pass
    elif 'libero' in eval_type:
        from run_libero_eval import evaluate_libero_policy

        if 'low_level' in eval_type:
            eval_type_libero = "libero_low_level_hard"
        elif 'conj' in eval_type:
            eval_type_libero = "libero_conj_hard"
        else:
            eval_type_libero = "libero_high_level_hard"

        new_rate, _, _ = evaluate_libero_policy(
            task_suite_name=eval_type_libero,
            num_trials_per_task=3,
            model_name=model_name,
            action_horizon=10,
            policy=policy,
            processors=processors,
            num_steps_wait=10,
            timestep=global_step
        )
    else:
        from run_calvin_eval import evaluate_policy
        from utils.calvin_utils import get_calvin_env

        env, calvin_cfg = get_calvin_env(
            train_cfg_path=None,
            merged_cfg_path=config_path,
            model=model_name,
            device_id=0,
        )

        if 'low_level' in eval_type:
            calvin_cfg.eval_type = 'train_low_random'
            tasks = OmegaConf.load(Path(__file__).parents[1] / "calvin/calvin_models/conf/callbacks/rollout/tasks/low_level_tasks.yaml")
            tasks_ann = OmegaConf.load(Path(__file__).parents[1] / "calvin/calvin_models/conf/annotations/low_level_tasks_validation.yaml")
        elif 'conj' in eval_type:
            calvin_cfg.eval_type = 'train_conjunction'
            tasks = OmegaConf.load(Path(__file__).parents[1] / "calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
            tasks_ann = OmegaConf.load(Path(__file__).parents[1] / "calvin/calvin_models/conf/annotations/low_level_tasks_validation.yaml")
        else:
            calvin_cfg.eval_type = 'train_high'
            tasks = OmegaConf.load(Path(__file__).parents[1] / "calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
            tasks_ann = OmegaConf.load(Path(__file__).parents[1] / "calvin/calvin_models/conf/annotations/new_playtable_validation.yaml")


        calvin_cfg.model = model_name
        calvin_cfg.action_horizon = 10
        num_sequences = 50

        _, new_rate, _, _ = evaluate_policy(
            cfg=calvin_cfg, 
            model=policy,
            processors=processors, 
            env=env, 
            task_oracle=tasks,
            annotations=tasks_ann,
            eval_sequences=None,
            counters=None,
            num_sequences=num_sequences
        )

    return new_rate



def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)



def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action = binarize_gripper_action(action)

    return action


def binarize_gripper_action(action):
    action[..., -1] = np.sign(action[..., -1])
    return action

def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action

def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den



task_categories = {
    "rotate_red_block_right": 1,
    "rotate_red_block_left": 1,
    "rotate_blue_block_right": 1,
    "rotate_blue_block_left": 1,
    "rotate_pink_block_right": 1,
    "rotate_pink_block_left": 1,
    "move_slider_left": 2,
    "move_slider_right": 2,
    "open_drawer": 3,
    "close_drawer": 3,
    "place_red_block_in_slider": 7,
    "place_blue_block_in_slider": 7,
    "place_pink_block_in_slider": 7,
    "place_red_block_in_drawer": 12,
    "place_blue_block_in_drawer": 12,
    "place_pink_block_in_drawer": 12,
    "stack_block": 10,
    "unstack_block": 11,
}


high_to_low_level_mappings = {
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

    "rotate_red_block_right": [
        ["grasp_red_block", "rotate_grasped_block_right", "ungrasp_block"],
    ],
    "rotate_red_block_left": [
        ["grasp_red_block", "rotate_grasped_block_left", "ungrasp_block"]
    ],
    "rotate_blue_block_right": [
        ["grasp_blue_block", "rotate_grasped_block_right", "ungrasp_block"]
    ],
    "rotate_blue_block_left": [
        ["grasp_blue_block", "rotate_grasped_block_left", "ungrasp_block"]
    ],
    "rotate_pink_block_right": [
        ["grasp_pink_block", "rotate_grasped_block_right", "ungrasp_block"]
    ],
    "rotate_pink_block_left": [
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
        ["grasp_red_block", 
         "place_grasped_block_over_table", "ungrasp_block"],
        ["grasp_blue_block",  
         "place_grasped_block_over_table", "ungrasp_block"],
        ["grasp_pink_block",  
         "place_grasped_block_over_table", "ungrasp_block"],
    ]
}

tasks = {
    # Rotate
    "rotate_red_block_right": [
        {"condition": {"red_block": "table"}, "effect": {"red_block": "table"}}
    ],
    "rotate_red_block_left": [
        {"condition": {"red_block": "table"}, "effect": {"red_block": "table"}}
    ],
    "rotate_blue_block_right": [
        {"condition": {"blue_block": "table"}, "effect": {"blue_block": "table"}}
    ],
    "rotate_blue_block_left": [
        {"condition": {"blue_block": "table"}, "effect": {"blue_block": "table"}}],
    "rotate_pink_block_right": [
        {"condition": {"pink_block": "table"}, "effect": {"pink_block": "table"}}
    ],
    "rotate_pink_block_left": [
        {"condition": {"pink_block": "table"}, "effect": {"pink_block": "table"}}
    ],

    # Slider/drawer move
    "move_slider_left": [
        {"condition": {"slider": "right"}, "effect": {"slider": "left"}}],
    "move_slider_right": [
        {"condition": {"slider": "left"}, "effect": {"slider": "right"}}
    ],
    "open_drawer": [
        {"condition": {"drawer": "closed"}, "effect": {"drawer": "open"}}
    ],
    "close_drawer": [
        {"condition": {"drawer": "open"}, "effect": {"drawer": "closed"}}
    ],

    "place_red_block_in_slider": [
        {
            "condition": {"red_block": "table", "slider": "right"},
            "effect": {"red_block": "slider_left"},
        },
        {
            "condition": {"red_block": "table", "slider": "left"},
            "effect": {"red_block": "slider_right"},
        },
        {
            "condition": {"red_block": "drawer", "drawer": "open", "slider": "right"},
            "effect": {"red_block": "slider_left"},
        },
        {
            "condition": {"red_block": "drawer", "drawer": "open", "slider": "left"},
            "effect": {"red_block": "slider_right"},
        },
    ],
    "place_blue_block_in_slider": [
        {
            "condition": {"blue_block": "table", "slider": "right"},
            "effect": {"blue_block": "slider_left"},
        },
        {
            "condition": {"blue_block": "table", "slider": "left"},
            "effect": {"blue_block": "slider_right"},
        },
        {
            "condition": {"blue_block": "drawer", "drawer": "open", "slider": "right"},
            "effect": {"blue_block": "slider_left"},
        },
        {
            "condition": {"blue_block": "drawer", "drawer": "open", "slider": "left"},
            "effect": {"blue_block": "slider_right"},
        },
    ],
    "place_pink_block_in_slider": [
        {
            "condition": {"pink_block": "table", "slider": "right"},
            "effect": {"pink_block": "slider_left"},
        },
        {
            "condition": {"pink_block": "table", "slider": "left"},
            "effect": {"pink_block": "slider_right"},
        },
        {
            "condition": {"pink_block": "drawer", "drawer": "open", "slider": "right"},
            "effect": {"pink_block": "slider_left"},
        },
        {
            "condition": {"pink_block": "drawer", "drawer": "open", "slider": "left"},
            "effect": {"pink_block": "slider_right"},
        },
    ],
    "place_red_block_in_drawer": [
        {
            "condition": {"red_block": "table", "drawer": "open"},
            "effect": {"red_block": "drawer"},
        },
        { 
            "condition": {"red_block": "slider_left", "slider": "right", "drawer": "open"},
            "effect": {"red_block": "drawer"},
        },
        { 
            "condition": {"red_block": "slider_right", "slider": "left", "drawer": "open"},
            "effect": {"red_block": "drawer"},
        },
    ],
    "place_blue_block_in_drawer": [
        {
            "condition": {"blue_block": "table", "drawer": "open"},
            "effect": {"blue_block": "drawer"},
        },
        { 
            "condition": {"blue_block": "slider_left", "slider": "right", "drawer": "open"},
            "effect": {"blue_block": "drawer"},
        },
        { 
            "condition": {"blue_block": "slider_right", "slider": "left", "drawer": "open"},
            "effect": {"blue_block": "drawer"},
        },
    ],
    "place_pink_block_in_drawer": [
        {
            "condition": {"pink_block": "table", "drawer": "open"},
            "effect": {"pink_block": "drawer"},
        },
        { 
            "condition": {"pink_block": "slider_left", "slider": "right", "drawer": "open"},
            "effect": {"pink_block": "drawer"},
        },
        { 
            "condition": {"pink_block": "slider_right", "slider": "left", "drawer": "open"},
            "effect": {"pink_block": "drawer"},
        },
    ],

    "stack_block": [
        {
            "condition": {"red_block": "table", "blue_block": "table", "pink_block": "drawer", "drawer": "closed"},
            "effect": {"red_block": "stacked", "blue_block": "stacked"},
        },
        {
            "condition": {"red_block": "table", "blue_block": "table", "pink_block": "slider_left", "slider": "left"},
            "effect": {"red_block": "stacked", "blue_block": "stacked"},
        },
        {
            "condition": {"red_block": "table", "blue_block": "table", "pink_block": "slider_right", "slider": "right"},
            "effect": {"red_block": "stacked", "blue_block": "stacked"},
        },
        {
            "condition": {"red_block": "table", "pink_block": "table", "blue_block": "drawer", "drawer": "closed"},
            "effect": {"red_block": "stacked", "pink_block": "stacked"},
        },
        {
            "condition": {"red_block": "table", "pink_block": "table", "blue_block": "slider_left", "slider": "left"},
            "effect": {"red_block": "stacked", "pink_block": "stacked"},
        },
        {
            "condition": {"red_block": "table", "pink_block": "table", "blue_block": "slider_right", "slider": "right"},
            "effect": {"red_block": "stacked", "pink_block": "stacked"},
        },
        {
            "condition": {"blue_block": "table", "pink_block": "table", "red_block": "drawer", "drawer": "closed"},
            "effect": {"blue_block": "stacked", "pink_block": "stacked"},
        },
        {
            "condition": {"blue_block": "table", "pink_block": "table", "red_block": "slider_left", "slider": "left"},
            "effect": {"blue_block": "stacked", "pink_block": "stacked"},
        },
        {
            "condition": {"blue_block": "table", "pink_block": "table", "red_block": "slider_right", "slider": "right"},
            "effect": {"blue_block": "stacked", "pink_block": "stacked"},
        },

    ],
    "unstack_block": [
        {
            "condition": {"red_block": "stacked", "blue_block": "stacked"},
            "effect": {"red_block": "table", "blue_block": "table"},
        },
        {
            "condition": {"red_block": "stacked", "pink_block": "stacked"},
            "effect": {"red_block": "table", "pink_block": "table"},
        },
        {
            "condition": {"blue_block": "stacked", "pink_block": "stacked"},
            "effect": {"blue_block": "table", "pink_block": "table"},
        }
    ]
}
