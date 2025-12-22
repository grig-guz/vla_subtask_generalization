
#!/bin/bash

WDB_ENTITY="grig_guz" 
WDB_PROJ="cogact" 
SEED=1
TRAIN_DATASET="libero_conj" # calvin_high_level, calvin_conj, calvin_low_level, libero_high_level, libero_conj, libero_low_level
CHECKPOINT_DIR="."
DATA_ROOT_DIR="."


# NOTE: The code automatically selects the appropriate number for gradient accumulation steps based on the values below.
GLOBAL_BATCH_SIZE=32
PER_DEVICE_BATCH_SIZE=16
NUM_GPUS=1

COGACT_CHECKPOINT="CogACT/CogACT-Large"


torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS  \
                        models/CogACT/scripts/wrapper_script.py \
                        --vla.max_steps 50000 \
                        --pretrained_checkpoint $COGACT_CHECKPOINT \
                        --vla.type prism-dinosiglip-224px+oxe+diffusion \
                        --vla.data_mix $TRAIN_DATASET \
                        --vla.expected_world_size $NUM_GPUS \
                        --vla.global_batch_size $GLOBAL_BATCH_SIZE \
                        --vla.per_device_batch_size $PER_DEVICE_BATCH_SIZE \
                        --vla.learning_rate 2e-5 \
                        --data_root_dir $DATA_ROOT_DIR \
                        --run_root_dir $CHECKPOINT_DIR \
                        --seed $SEED \
                        --run_id "cogact_$TRAIN_DATASET""_$SEED" \
                        --image_aug True \
                        --wandb_project $WDB_PROJ \
                        --wandb_entity $WDB_ENTITY \
                        --save_interval 5000 \
                        --repeated_diffusion_steps 8 \
                        --future_action_window_size 15 \
                        --action_model_type DiT-L \
                        --is_resume False