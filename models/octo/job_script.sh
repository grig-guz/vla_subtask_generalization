#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=ubcml
#SBATCH --job-name=repr
#SBATCH --mem=96GB
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=6
#SBATCH --time=47:59:59
#SBATCH --exclude=ubc-ml[01,06]
################################################################################

source /ubc/cs/research/ubc_ml/gguz/miniconda3/etc/profile.d/conda.sh
conda activate octo-env
cd /ubc/cs/research/nlp/grigorii/projects/octo_original/octo

python scripts/finetune.py                                             --config.random_init=False                                             --config.pretrained_path=hf://rail-berkeley/octo-small-1.5                                             --config.dataset_kwargs.name=medium_level_tasks_dataset_no_paraphrase_octo                                             --config.dataset_kwargs.data_dir=/ubc/cs/research/ubc_ml/gguz/datasets/                                             --config.calvin_config_path=/ubc/cs/research/nlp/grigorii/projects/openvla/experiments/robot/calvin/conf/med_tasks_config.yaml                                             --config.traj_transform_kwargs.action_horizon=4                                             --config.window_size=2                                             --config.batch_size=32                                             --config.save_dir=/ubc/cs/research/ubc_ml/gguz/exp_data                                             --config.save_interval=10000                                             --config.num_steps=800000                                             --config.video_save_dir=/ubc/cs/research/nlp/grigorii/projects/octo_original/octo/experiments/video_saves                                             --config.optimizer.grad_accumulation_steps=8                                             --video_save_dir=/ubc/cs/research/nlp/grigorii/projects/octo_original/octo/experiments/video_saves                                             --config.jax_compilation_path=~/                                             --config.seed=0