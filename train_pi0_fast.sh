
#!/bin/bash

TRAIN_DATASET="calvin_conj" # calvin_high_level, calvin_conj, calvin_low_level, libero_high_level, libero_conj, libero_low_level 

CHECKPOINT_DIR="/home/gguz/scratch/testing/"
DATA_ROOT_DIR="./datasets"

SEED=5
GLOBAL_BATCH_SIZE=128
NUM_GPUS=4

cd models/openpi
source .venv/bin/activate

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $TRAIN_DATASET \
        --exp-name=my_experiment_low_$SEED \
        --seed=$SEED \
        --overwrite \
        --checkpoint-base-dir=$CHECKPOINT_DIR \
        --batch-size=$GLOBAL_BATCH_SIZE \
        --fsdp-devices=$NUM_GPUS \
        --save-interval=5000 \
        --num-train-steps=50000
