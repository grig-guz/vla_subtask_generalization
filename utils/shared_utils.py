import os

import os
import sys
import threading
import shutil
import torch
import numpy as np
import random 
import tensorflow as tf
from pathlib import Path
from omegaconf import OmegaConf




def run_validation(policy, eval_type, model_name, global_step, config_path="utils/med_tasks_config.yaml"):
    from run_calvin_eval import evaluate_policy
    from run_libero_eval import evaluate_libero_policy
    from utils.calvin_utils import get_calvin_env

    if 'libero' in eval_type:
        new_rate, _ = evaluate_libero_policy(
            task_suite_name=eval_type,
            num_trials_per_task=3,
            model_name=model_name,
            action_horizon=10,
            policy=policy,
            num_steps_wait=10,
            timestep=global_step
        )
    else:
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
            processor=None, 
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

def resize_image(img, resize_size, primary=True):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)

    # If the primary camera image was shifted/scaled in Octo 
    # (OpenVLA code already handles this)
    if primary:
        avg_scale = 0.9
        avg_ratio = 1.0
        new_height = tf.clip_by_value(tf.sqrt(avg_scale / avg_ratio), 0, 1)
        new_width = tf.clip_by_value(tf.sqrt(avg_scale * avg_ratio), 0, 1)
        height_offset = (1 - new_height) / 2
        width_offset = (1 - new_width) / 2
        bounding_box = tf.stack(
            [
                height_offset,
                width_offset,
                height_offset + new_height,
                width_offset + new_width,
            ],
        )
        img = tf.image.crop_and_resize(
            img[None], bounding_box[None], [0], resize_size
        )[0]

        
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_action(cfg, model, obs, task_label, processor=None):
    """Queries the model to get an action."""
    if cfg.model_family == "openvla":
        action = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
        assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action

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
        action[..., -1] = np.sign(action[..., -1])

    return action

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




tasks_low_level = {
    # Description: Initially, the robot should not be holding anything. At the end, it should be holding the corresponding object.
    "grasp_red_block": [
        {"condition": {"red_block": "table", "grasped": 0, "contact": 0}, "effect": {"grasped": "red_block"}},
        {"condition": {"red_block": "drawer", "drawer": "open", "grasped": 0, "contact": 0}, "effect": {"grasped": "red_block"}},
        {"condition": {"red_block": "slider_left", "slider": "right", "grasped": 0, "contact": 0}, "effect": {"grasped": "red_block"}},
        {"condition": {"red_block": "slider_right", "slider": "left", "grasped": 0, "contact": 0}, "effect": {"grasped": "red_block"}},
    ],
    "grasp_blue_block": [
        {
            "condition": {"blue_block": "table", "grasped": 0, "contact": 0}, 
            "effect": {"grasped": "blue_block"}
         },
        {
            "condition": {"blue_block": "drawer", "drawer": "open", "grasped": 0, "contact": 0}, 
            "effect": {"grasped": "blue_block"}
        },
        {
            "condition": {"blue_block": "slider_left", "slider": "right", "grasped": 0, "contact": 0}, 
            "effect": {"grasped": "blue_block"}
        },
        {"condition": {"blue_block": "slider_right", "slider": "left", "grasped": 0, "contact": 0}, 
         "effect": {"grasped": "blue_block"}},
    ],
    "grasp_pink_block": [
        {
            "condition": {"pink_block": "table", "grasped": 0, "contact": 0}, 
            "effect": {"grasped": "pink_block"}
        },
        {
            "condition": {"pink_block": "drawer", "drawer": "open", "grasped": 0, "contact": 0}, 
            "effect": {"grasped": "pink_block"}
        },
        {
            "condition": {"pink_block": "slider_left", "slider": "right", "grasped": 0, "contact": 0}, 
            "effect": {"grasped": "pink_block"}
        },
        {
            "condition": {"pink_block": "slider_right", "slider": "left", "grasped": 0, "contact": 0}, 
            "effect": {"grasped": "pink_block"}
        },
    ],
    "grasp_slider": [{"condition": {"grasped": 0, "contact": 0}, "effect": {"grasped": "slider"}}],
    "grasp_drawer": [{"condition": {"grasped": 0, "contact": 0}, "effect": {"grasped": "drawer"}}],


    # Description: Initially, the robot should be holding an object. At the end, it shouldn't be holding anything.
    "ungrasp_block": [{"condition": {"grasped": ["red_block", "blue_block", "pink_block"], "contact": 0}, 
                       "effect": {"grasped": 0, "red_block_lifted": 0, "blue_block_lifted": 0, "pink_block_lifted": 0}
                       }],
    "ungrasp_slider": [{"condition": {"grasped": "slider", "contact": 0}, "effect": {"grasped": 0}}],
    "ungrasp_drawer": [{"condition": {"grasped": "drawer", "contact": 0}, "effect": {"grasped": 0}}],


    # Gripper placement
    "place_grasped_block_over_red_block": [
        {"condition": {"red_block": "table", 'blue_block_lifted': 1, "contact": 0}, "effect": {'blue_block': 'red_block'}},
        {"condition": {"red_block": "table", 'pink_block_lifted': 1, "contact": 0}, "effect": {'pink_block': 'red_block'}},
    ],
    "place_grasped_block_over_blue_block": [
        {"condition": {"blue_block": "table", 'red_block_lifted': 1, "contact": 0}, "effect": {'red_block': 'blue_block'}},
        {"condition": {"blue_block": "table", 'pink_block_lifted': 1, "contact": 0}, "effect": {'pink_block': 'blue_block'}},

    ],
    "place_grasped_block_over_pink_block": [
        {"condition": {"pink_block": "table", 'red_block_lifted': 1, "contact": 0}, "effect": {'red_block': 'pink_block'}},
        {"condition": {"pink_block": "table", 'blue_block_lifted': 1, "contact": 0}, "effect": {'blue_block': 'pink_block'}},
    ],

    "move_slider_left": [{"condition": {"slider": "right", "grasped": "slider", "contact": 0}, "effect": {"slider": "left"}}],
    "move_slider_right": [{"condition": {"slider": "left", "grasped": "slider", "contact": 0}, "effect": {"slider": "right"}}],

    "open_drawer": [{"condition": {"drawer": "closed", "grasped": "drawer", "contact": 0}, "effect": {"drawer": "open"}}],
    "close_drawer": [{"condition": {"drawer": "open", "grasped": "drawer", "contact": 0}, "effect": {"drawer": "closed"}}],

    # lifting
    "lift_grasped_block": [
        {"condition": {"red_block": "table", "grasped": "red_block", "red_block_lifted": 0, "contact": 0}, "effect": {"red_block_lifted": 1}},
        {"condition": {"blue_block": "table", "grasped": "blue_block", "blue_block_lifted": 0, "contact": 0}, "effect": {"blue_block_lifted": 1   }},
        {"condition": {"pink_block": "table", "grasped": "pink_block", "pink_block_lifted": 0, "contact": 0}, "effect": {"pink_block_lifted": 1}},
        {"condition": {"red_block": "slider_left", "grasped": "red_block", "red_block_lifted": 0, "contact": 0}, "effect": {"red_block_lifted": 1}},
        {"condition": {"blue_block": "slider_left", "grasped": "blue_block", "blue_block_lifted": 0, "contact": 0}, "effect": { "blue_block_lifted": 1}},
        {"condition": {"pink_block": "slider_left", "grasped": "pink_block", "pink_block_lifted": 0, "contact": 0}, "effect": { "pink_block_lifted": 1}},
        {"condition": {"red_block": "slider_right", "grasped": "red_block", "red_block_lifted": 0, "contact": 0}, "effect": { "red_block_lifted": 1}},
        {"condition": {"blue_block": "slider_right", "grasped": "blue_block", "blue_block_lifted": 0, "contact": 0}, "effect": { "blue_block_lifted": 1}},
        {"condition": {"pink_block": "slider_right", "grasped": "pink_block", "pink_block_lifted": 0, "contact": 0}, "effect": { "pink_block_lifted": 1}},
        {"condition": {"red_block": "drawer", "grasped": "red_block", "red_block_lifted": 0, "contact": 0}, "effect": { "red_block_lifted": 1}},
        {"condition": {"blue_block": "drawer", "grasped": "blue_block", "blue_block_lifted": 0, "contact": 0}, "effect": { "blue_block_lifted": 1}},
        {"condition": {"pink_block": "drawer", "grasped": "pink_block", "pink_block_lifted": 0, "contact": 0}, "effect": {"pink_block_lifted": 1}}
    ],

    # rotation
    "rotate_grasped_block_right": [
        {"condition": {"red_block": "table", "grasped": "red_block", "contact": 0}, "effect": {"red_block": "table", "red_block_lifted": 1}},
        {"condition": {"blue_block": "table", "grasped": "blue_block", "contact": 0}, "effect": {"blue_block": "table", "blue_block_lifted": 1}},
        {"condition": {"pink_block": "table", "grasped": "pink_block", "contact": 0}, "effect": {"pink_block": "table", "pink_block_lifted": 1}},
    ],

    "rotate_grasped_block_left": [
        {"condition": {"red_block": "table", "grasped": "red_block", "contact": 0}, "effect": {"red_block": "table", "red_block_lifted": 1}},
        {"condition": {"blue_block": "table", "grasped": "blue_block", "contact": 0}, "effect": {"blue_block": "table", "blue_block_lifted": 1}},
        {"condition": {"pink_block": "table", "grasped": "pink_block", "contact": 0}, "effect": {"pink_block": "table", "pink_block_lifted": 1}},
    ],

    # open/close (slider, drawer) -> grasp slider/drawer + 
    "place_grasped_block_over_drawer": [        
        {
            "condition": {"red_block": ["table", "slider_right", "slider_left", "blue_block", "pink_block"], "drawer": "open", "grasped": "red_block", 'red_block_lifted': 1, "contact": 0},
            "effect": {"red_block": "drawer", "grasped": "red_block"},
        },
        {
            "condition": {"blue_block": ["table", "slider_right", "slider_left", "red_block", "pink_block"], "drawer": "open", "grasped": "blue_block", 'blue_block_lifted': 1, "contact": 0},
            "effect": {"blue_block": "drawer", "grasped": "blue_block"},
        },
        {
            "condition": {"pink_block": ["table", "slider_right", "slider_left", "red_block", "blue_block"], "drawer": "open", "grasped": "pink_block", 'pink_block_lifted': 1, "contact": 0},
            "effect": {"pink_block": "drawer", "grasped": "pink_block"},
        },
    ],

    "place_grasped_block_over_slider": [
        {
            "condition": {"red_block": ["table", "drawer", "blue_block", "pink_block"], "slider": "right", "grasped": "red_block", 'red_block_lifted': 1, "contact": 0},
            "effect": {"red_block": "slider_left"},
        },
        {
            "condition": {"red_block": ["table", "drawer", "blue_block", "pink_block"], "slider": "left", "grasped": "red_block", 'red_block_lifted': 1, "contact": 0},
            "effect": {"red_block": "slider_right"},
        },
        {
            "condition": {"blue_block": ["table", "drawer", "red_block", "pink_block"], "slider": "right", "grasped": "blue_block", 'blue_block_lifted': 1, "contact": 0},
            "effect": {"blue_block": "slider_left"},
        },
        {
            "condition": {"blue_block": ["table", "drawer", "red_block", "pink_block"], "slider": "left", "grasped": "blue_block", 'blue_block_lifted': 1, "contact": 0},
            "effect": {"blue_block": "slider_right"},
        },
        {
            "condition": {"pink_block": ["table", "drawer", "red_block", "blue_block"], "slider": "right", "grasped": "pink_block", 'pink_block_lifted': 1, "contact": 0},
            "effect": {"pink_block": "slider_left"},
        },
        {
            "condition": {"pink_block": ["table", "drawer", "red_block", "blue_block"], "slider": "left", "grasped": "pink_block", 'pink_block_lifted': 1, "contact": 0},
            "effect": {"pink_block": "slider_right"},
        },
    ],

    "place_grasped_block_over_table": [
        {
            "condition": {"red_block": ["drawer", "slider_right", "slider_left", "blue_block", "pink_block"], "grasped": "red_block", 'red_block_lifted': 1, "contact": 0},
            "effect": {"red_block": "table"},
        },
        {
            "condition": {"blue_block": ["drawer", "slider_right", "slider_left", "red_block", "pink_block"], "grasped": "blue_block", 'blue_block_lifted': 1, "contact": 0},
            "effect": {"blue_block": "table"},
        },
        {
            "condition": {"pink_block": ["drawer", "slider_right", "slider_left", "red_block", "blue_block"], "grasped": "pink_block", 'pink_block_lifted': 1, "contact": 0},
            "effect": {"pink_block": "table"},
        },
    ],
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
