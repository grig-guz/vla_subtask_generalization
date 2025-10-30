
TASK_SUITE_NAME="libero_low_level" # libero_low_level, libero_low_level_hard
MODEL="pi0_fast" # octo, pi0_fast, cogact
PRETRAINED_CHECKPOINT="."
RESULTS_SAVE_DIR="."
VIDEO_SAVE_DIR="."

python run_libero_eval.py \
    --task_suite_name=$TASK_SUITE_NAME \
    --model=$MODEL \
    --pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    --results_save_dir=$RESULTS_SAVE_DIR \
    --video_save_dir=$VIDEO_SAVE_DIR     
