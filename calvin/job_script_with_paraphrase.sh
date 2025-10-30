#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=ubcml
#SBATCH --job-name=with_para
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:59:59
#SBATCH --exclude=ubc-ml[01-05,07]

################################################################################

source /ubc/cs/research/ubc_ml/gguz/miniconda3/etc/profile.d/conda.sh
conda activate calvin_venv
cd /ubc/cs/research/nlp/grigorii/projects/calvin/calvin_models/calvin_agent/utils

srun python automatic_lang_annotator_mp.py lang_folder=lang_annotations_med_level_with_paraphrase