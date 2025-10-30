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
import sys
sys.path.append("/scratch/ssd004/scratch/gguz/projects/low_level_tasks/models/openpi")

#from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
from pathlib import Path

#REPO_NAME = "your_hf_username/libero"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_high_level",
    "libero_conj",
    "libero_low_level",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    for REPO_NAME in RAW_DATASET_NAMES:

        output_path = Path("/scratch/ssd004/scratch/gguz/datasets/libero_lerobot/" + REPO_NAME)
        if output_path.exists():
            shutil.rmtree(output_path)

        # Create LeRobot dataset, define features to store
        # OpenPi assumes that proprio is stored in `state` and actions in `action`
        # LeRobot assumes that dtype of image data is `image`
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            robot_type="panda",
            fps=20,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (128, 128, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float64",
                    "shape": (9,),
                    "names": ["state"],
                },
                "actions": {
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
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode()
                    }
                )
            dataset.save_episode()

        # Consolidate the dataset, skip computing stats since we will do that later

        # Optionally push to the Hugging Face Hub


if __name__ == "__main__":
    tyro.cli(main)
