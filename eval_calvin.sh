

EVAL_TYPE="high_random" # high_random, conj_random, low_random, high_random_easy_eval, conj_random_easy_eval, low_random_easy_eval
MODEL="pi05" # octo, pi0_fast, cogact
PRETRAINED_CHECKPOINT="/home/gguz/scratch/checkpoints/pi05/calvin_high_1/checkpoints/last"



RESULTS_SAVE_DIR="/home/gguz/scratch/results"
VIDEO_SAVE_DIR="/home/gguz/scratch/results/videos"
CALVIN_CONFIG_PATH="./utils/med_tasks_config.yaml"

cd /home/gguz/projects/aip-vshwartz/gguz/vla_subtask_generalization
uv run run_calvin_eval.py \
    --eval_type=$EVAL_TYPE \
    --model=$MODEL \
    --pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    --results_save_dir=$RESULTS_SAVE_DIR \
    --video_save_dir=$VIDEO_SAVE_DIR \
    --calvin_config_path=$CALVIN_CONFIG_PATH \
    --num_videos=10 \
    --num_sequences=1000
