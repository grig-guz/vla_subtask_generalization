
#!/bin/bash

CHECKPOINT_DIR="/home/gguz/scratch/checkpoints"
DATA_ROOT_DIR="/home/gguz/projects/aip-vshwartz/gguz/vla_subtask_generalization/datasets"
CALVIN_CONFIG_PATH="./utils/med_tasks_config.yaml"
SEED=1


TRAIN_DATASET="calvin_high_level" # calvin_high_level, calvin_conj, calvin_low_level, libero_high_level, libero_conj, libero_low_level

RANDOM_INIT=False # whether to initialize the model weights randomly (instead of using pretrained weights)
MODEL_SIZE=small # small or base
ACTION_HORIZON=10 # action chunk size
WINDOW_SIZE=2 # number of image frames to use as input (if 2: frame at timestep t (current) and t-1) 
DO_SUBSAMPLING=False # whether to select a random subsample of the training dataset (for CALVIN high level only)
ANN_FILE_PATH="./utils/bootstrap_lang_ann.py" # for bootstrap experiments

# NOTE: jax automatically detects the number of GPUs available and splits the batch so the batch size on each GPU is GLOBAL_BATCH_SIZE/num_gpus 
GLOBAL_BATCH_SIZE=16

# NOTE: adjust num_steps and save_interval correspondingly if increasing grad_accumulation_steps. 
GRADIENT_ACCUMULATION_STEPS=1

python models/octo/scripts/finetune.py \
        --config.random_init=$RANDOM_INIT \
        --config.pretrained_path=hf://rail-berkeley/octo-$MODEL_SIZE-1.5 \
        --config.dataset_kwargs.name=$TRAIN_DATASET \
        --config.dataset_kwargs.data_dir=$DATA_ROOT_DIR \
        --config.calvin_config_path=$CALVIN_CONFIG_PATH \
        --config.traj_transform_kwargs.action_horizon=$ACTION_HORIZON \
        --config.window_size=$WINDOW_SIZE \
        --config.batch_size=$GLOBAL_BATCH_SIZE \
        --config.save_dir=$CHECKPOINT_DIR \
        --config.save_interval=5000 \
        --config.num_steps=50000 \
        --config.optimizer.grad_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
        --config.jax_compilation_path="." \
        --config.seed=$SEED \
        --config.ann_file_path=$ANN_FILE_PATH \
        --config.do_subsampling=$DO_SUBSAMPLING
