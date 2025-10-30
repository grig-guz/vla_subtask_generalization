#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=ubcml
#SBATCH --job-name=jupyter
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --time=8:0:0
#SBATCH --exclude=ubc-ml[02]

################################################################################

source /ubc/cs/research/ubc_ml/gguz/miniconda3/etc/profile.d/conda.sh
conda activate octo-env
cd /ubc/cs/research/nlp/grigorii/projects/octo-original/octo
srun jupyter-notebook --no-browser --ip=0.0.0.0
