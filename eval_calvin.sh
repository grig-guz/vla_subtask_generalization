

EVAL_TYPE="high_random" # high_random, conj_random, low_random, high_random_easy_eval, conj_random_easy_eval, low_random_easy_eval
MODEL="pi0_fast" # octo, pi0_fast, cogact
PRETRAINED_CHECKPOINT="."
RESULTS_SAVE_DIR="."
VIDEO_SAVE_DIR="."
CALVIN_CONFIG_PATH="."

python run_calvin_eval.py \
    --eval_type=$EVAL_TYPE \
    --model=$MODEL \
    --pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    --results_save_dir=$RESULTS_SAVE_DIR \
    --video_save_dir=$VIDEO_SAVE_DIR \
    --calvin_config_path=$CALVIN_CONFIG_PATH 
