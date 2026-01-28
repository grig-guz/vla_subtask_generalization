"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
from pathlib import Path
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
import sys
import numpy as np
sys.path.append("/home/gguz/projects/aip-vshwartz/gguz/vla_subtask_generalization/")
sys.path.append("/home/gguz/links/scratch/vla_subtask_generalization")

from utils.calvin_utils import get_calvin_env


REPO_NAMES = [#"calvin_high_level_wrist", 
              "libero_low_level_single",
              "libero_high_level_single", 
              "libero_conj_single", 
              #"calvin_conj", 
              ]


def main(data_dir: str, *, push_to_hub: bool = False):
    #calvin_env, _ = get_calvin_env(
    #        train_cfg_path=None,
    #        merged_cfg_path="/home/gguz/projects/aip-vshwartz/gguz/vla_subtask_generalization/utils/med_tasks_config.yaml",
    #        model='',
    #        device_id=0,
    #)

    for REPO_NAME in REPO_NAMES:

        # Create LeRobot dataset, define features to store
        # OpenPi assumes that proprio is stored in `state` and actions in `action`
        # LeRobot assumes that dtype of image data is `image`
        if 'calvin' in REPO_NAME:
            state_dim = 15
        else:
            state_dim = 8
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            robot_type="panda",
            fps=30,
            features={
                "observation.images.camera1": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.camera2": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float64",
                    "shape": (state_dim,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float64",
                    "shape": (7,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

        # Loop over raw Libero datasets and write episodes to the LeRobot dataset
        # You can modify this for your own data format
        raw_dataset = tfds.load(REPO_NAME, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                if 'calvin' in REPO_NAME:
                    calvin_env.reset(robot_obs=step["observation"]["state"], scene_obs=step["observation"]["scene_state"])
                    viz = calvin_env.get_obs()
                    wrist_cam = viz['rgb_obs']['rgb_gripper']
                else:
                    wrist_cam = step["observation"]["wrist_image"]
                dataset.add_frame({
                        "observation.images.camera1": step["observation"]["image"],
                        "observation.images.camera2": np.zeros((256, 256, 3)),#wrist_cam,#np.zeros((256, 256, 3)),#wrist_cam,#step["observation"]["wrist_image"],
                        #'observation.images.camera2_is_pad': True,
                        "observation.state": step["observation"]["state"],
                        "action": step["action"],
                        "task": step["language_instruction"].decode()
                    }
                )
            dataset.save_episode()

        # Consolidate the dataset, skip computing stats since we will do that later
        #dataset.consolidate(run_compute_stats=False)

        # Optionally push to the Hugging Face Hub
        if push_to_hub:
            dataset.push_to_hub(
                tags=["libero", "panda", "rlds"],
                private=False,
                push_videos=True,
                license="apache-2.0",
            )


if __name__ == "__main__":
    tyro.cli(main)
