#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=ubcml
#SBATCH --job-name=no_para
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:59:59
#SBATCH --exclude=ubc-ml[01-05,07]

################################################################################

source /ubc/cs/research/ubc_ml/gguz/miniconda3/etc/profile.d/conda.sh
conda activate calvin_venv

cd /ubc/cs/research/nlp/grigorii/projects/low_level_tasks

srun python calvin/calvin_models/calvin_agent/utils/automatic_lang_annotator_mp.py trainer.devices=1