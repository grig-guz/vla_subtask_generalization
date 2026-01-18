import numpy as np
import torch

import hydra
from omegaconf import OmegaConf

from pathlib import Path
import os

import logging
from pathlib import Path
from typing import Dict

import cv2
import json

import torch
import tensorflow as tf
import yaml

from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import torch

from calvin_env.envs.play_table_env import get_env
from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id
import time
from collections import Counter
from utils.shared_utils import temp_seed


logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})




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


def get_libero_env(task, resolution=256):
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_env_and_checkpoint(cfg):
    env, calvin_cfg = get_calvin_env(
        cfg.pretrained_checkpoint,
        cfg.calvin_config_path,
        cfg.model,
        device_id=0,
    )

    if cfg.model == "octo":
        model, processor, calvin_cfg = load_octo_checkpoint(cfg.pretrained_checkpoint, cfg, calvin_cfg)
        calvin_cfg.action_horizon = 10
    elif cfg.model == 'pi0_fast':
        model, processor = load_pi0_fast_checkpoint(cfg.pretrained_checkpoint)
        calvin_cfg.action_horizon = 10
    elif cfg.model == 'cogact':
        model, processor = load_cogact_checkpoint(cfg.pretrained_checkpoint)
        calvin_cfg.action_horizon = 10
    elif cfg.model in ['smolvla', 'groot']:
        model, processor = load_smolvla_groot_checkpoint(cfg.pretrained_checkpoint)
        calvin_cfg.action_horizon = 10


    calvin_cfg.ep_len = 360

    return model, processor, env, calvin_cfg

def load_octo_checkpoint(checkpoint_path, cfg, calvin_cfg):

    from models.octo.octo.model.octo_model import OctoModel

    model = OctoModel.load_pretrained(checkpoint_path)
    processor = None
    with open(Path(checkpoint_path) / 'finetune_config.json', 'r') as file:
        finetune_config = json.load(file)
    if calvin_cfg != None:
        calvin_cfg.image_obs_keys = finetune_config['dataset_kwargs']['image_obs_keys']
    return model, processor, calvin_cfg

def load_smolvla_groot_checkpoint(checkpoint_path):
    
    from lerobot.policies.factory import make_pre_post_processors

    if 'smolvla' in checkpoint_path:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy_cls = SmolVLAPolicy
    elif 'groot' in checkpont_path:
        from lerobot.policies.groot.groot_n1 import GR00TN15
        policy_cls = GR00TN15


    policy = policy_cls.from_pretrained(checkpoint_path).to("cuda").eval()
    policy.config.n_action_steps = 10
    
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": {}},
    }

    processors = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    return policy, processors



def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action    

def load_pi0_fast_checkpoint(pretrained_path):
    from openpi.training import config
    from openpi.policies import policy_config

    if 'libero' in pretrained_path:
        if 'high_level' in pretrained_path:
            config = config.get_config("libero_high_level")
        elif 'conj' in pretrained_path:
            config = config.get_config("libero_conj")
        else:
            config = config.get_config("libero_low_level")
    else:
        if 'high_level' in pretrained_path:
            config = config.get_config("calvin_high_level")
        elif 'conj' in pretrained_path:
            config = config.get_config("calvin_conj")
        else:
            config = config.get_config("calvin_low_level")

    policy = policy_config.create_trained_policy(config, pretrained_path)

    return policy, None

def load_cogact_checkpoint(pretrained_checkpoint, model_state_dict=None):
    from models.CogACT.scripts.deploy import CogACTService
    if 'libero' in pretrained_checkpoint:
        if 'conj' in pretrained_checkpoint:
            unnorm_key = "libero_conj"
        elif 'low_level' in pretrained_checkpoint:
            unnorm_key = "libero_low_level"
        elif 'high_level' in pretrained_checkpoint:
            unnorm_key = "libero_high_level"
        else:
            raise Exception("unknown unnorm_key")
    else:
        if 'conj' in pretrained_checkpoint:
            unnorm_key = "calvin_conj"
        elif 'low_level' in pretrained_checkpoint:
            unnorm_key = "calvin_low_level"
        elif 'high_level' in pretrained_checkpoint:
            unnorm_key = "calvin_high_level"
        else:
            raise Exception("unknown unnorm_key")

    policy = CogACTService(
        saved_model_path=pretrained_checkpoint,
        unnorm_key=unnorm_key,
        action_model_type="DiT-L",  # choose from ['DiT-Small', 'DiT-Base', 'DiT-Large'] to match the model weight
        use_bf16=False,
        action_chunking_window=10,
        action_ensemble=False,
        action_chunking=True,
        model_state_dict=model_state_dict
    )
    return policy, None



def get_env_state_for_initial_condition(initial_condition, seeds_dict):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    # we want to have a "deterministic" random seed for each initial condition
    if seeds_dict:
        init_state_idx = []
        for key, value in initial_condition.items():
            init_state_idx.append(key)
            init_state_idx.append(value)

        seed = seeds_dict[tuple(init_state_idx)]
    else:
        seed = hash(str(initial_condition.values())) % 500

    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        scene_obs[1] = 0.0
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
            
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs



def join_vis_lang(img, lang_text):
    """Takes as input an image and a language instruction and visualizes them with cv2"""
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (500, 500))
    add_text(img, lang_text)
    cv2.imshow("simulation cam", img)
    cv2.waitKey(1)

def get_calvin_env(train_cfg_path, merged_cfg_path, model, device_id=0):
    merged_cfg_path = Path(merged_cfg_path)
    merged_cfg = OmegaConf.load(merged_cfg_path)
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")

    device = torch.device(f"cuda:{device_id}")
    env = HulcWrapper(config_dir=Path(__file__).parents[0] / 'render_conf.yaml', device=device, use_egl=True, show_gui=False)

    return env, merged_cfg

def get_calvin_cfg(merged_cfg_path):
    merged_cfg_path = Path(merged_cfg_path)
    merged_cfg = OmegaConf.load(merged_cfg_path)
    return merged_cfg



class HulcWrapper(gym.Wrapper):
    def __init__(self, config_dir, device, use_egl, show_gui=False, **kwargs):
        self.set_egl_device(device)
        env = get_env(
            config_dir, show_gui=show_gui, obs_space=None, use_egl=use_egl, **kwargs
        )
        super(HulcWrapper, self).__init__(env)
        #self.observation_space_keys = dataset_loader.observation_space
        #self.transforms = dataset_loader.transforms
        #self.proprio_state = dataset_loader.proprio_state
        self.device = device
        self.relative_actions = True
        logger.info(f"Initialized PlayTableEnv for device {self.device}")

    @staticmethod
    def set_egl_device(device):
        if "EGL_VISIBLE_DEVICES" in os.environ:
            logger.warning("Environment variable EGL_VISIBLE_DEVICES is already set. Is this intended?")
        cuda_id = device.index if device.type == "cuda" else 0
        try:
            egl_id = get_egl_device_id(cuda_id)
        except EglDeviceNotFoundError:
            logger.warning(
                "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
                "When using DDP with many GPUs this can lead to OOM errors. "
                "Did you install PyBullet correctly? Please refer to calvin env README"
            )
            egl_id = 0
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")

    def step(
        self, action_tensor: torch.Tensor, use_angles=False
    ) -> Tuple[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], int, bool, Dict]:
        if use_angles:
            action = action_tensor
        elif self.relative_actions:
            action = action_tensor
            assert len(action) == 7
        else:
            if action_tensor.shape[-1] == 7:
                slice_ids = [3, 6]
            elif action_tensor.shape[-1] == 8:
                slice_ids = [3, 7]
            else:
                logger.error("actions are required to have length 8 (for euler angles) or 9 (for quaternions)")
                raise NotImplementedError
            action = np.split(action_tensor.squeeze().cpu().detach().numpy(), slice_ids)
        action[-1] = 1 if action[-1] > 0 else -1
        start = time.time()
        o, r, d, i = self.env.step(action, use_angles)
        start = time.time()

        return o, r, d, i

    def reset(
        self,
        reset_info: Dict[str, Any] = None,
        batch_idx: int = 0,
        seq_idx: int = 0,
        scene_obs: Any = None,
        robot_obs: Any = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        if reset_info is not None:
            obs = self.env.reset(
                robot_obs=reset_info["robot_obs"][batch_idx, seq_idx],
                scene_obs=reset_info["scene_obs"][batch_idx, seq_idx],
            )
        elif scene_obs is not None or robot_obs is not None:
            obs = self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
        else:
            obs = self.env.reset()

        return obs

    def get_info(self):
        return self.env.get_info()

    def get_obs(self):
        obs = self.env.get_obs()
        return obs


def add_text(img, lang_text, loc='bot'):
    height, width, _ = img.shape
    if lang_text != "":
        if loc == 'bot':
            coord = (1, int(height - 10))
        elif loc == 'top':
            coord = (1, 10)
        else:
            raise Exception("Unknown location of text to be drawn at")

        font_scale = (0.7 / 500) * width
        thickness = 1
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=thickness * 3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def format_sftp_path(path):
    """
    When using network mount from nautilus, format path
    """
    if path.as_posix().startswith("sftp"):
        uid = os.getuid()
        path = Path(f"/run/user/{uid}/gvfs/sftp:host={path.as_posix()[6:]}")
    return path
