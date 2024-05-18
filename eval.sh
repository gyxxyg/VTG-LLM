#!/bin/bash

DIR="yourpath/VTG-LLM"
MODEL_DIR="yourpath/checkpoint_9.pth"


# TASK='dvc'
# ANNO_DIR='yourpath/data/dense_video_captioning/youcook2'
# VIDEO_DIR='yourpath/data/dense_video_captioning/youcook2/YouCook2_asr_denseCap/youcook2_6fps_224'
# DATASET='youcook'
# SPLIT='val'
# PROMPT_FILE="prompts/${TASK}_description_zeroshot_time_token_fmt_v2.txt"
# GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"
# ASR_DIR='data/YouCook2-BB/YouCook2_asr_denseCap/whisper_outputs_with_time/small.en.cleaned/'

# TASK='dvc'
# ANNO_DIR='yourpath/data/dense_video_captioning/anet'
# VIDEO_DIR='/group/20009/data/ActivityNet/anet_6fps_224'
# DATASET='activitynet'
# SPLIT='test'
# PROMPT_FILE="yourpath/VTG-LLM/prompts/dvc_description_zeroshot_time_token_fmt_anet.txt"
# GT_FILE="yourpath/data/dense_video_captioning/anet/val_2.json"

# TASK='tvg'
# ANNO_DIR="/cfs/cfs-lugcocyb/messijyliu/data/TimeIT/data/temporal_video_grounding/charades/charades_annotation"
# VIDEO_DIR="/cfs/cfs-lugcocyb/messijyliu/data/Charades/videos"
# DATASET='charades'
# SPLIT='test'
# PROMPT_FILE="prompts/${TASK}_description_zeroshot_time_token_fmt_v3.txt"
# GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"
# ASR_DIR='data/Charades/whisper_outputs_with_te/tiny.en.cleaned/'

# TASK='vhd'
# ANNO_DIR='yourpath/data/video_highlight_detection/qvhighlights'
# VIDEO_DIR='/cfs/cfs-lugcocyb/messijyliu/data/QVhighlights/videos/val'
# DATASET='qvhighlights'
# SPLIT='val'
# PROMPT_FILE="yourpath/VTG-LLM/prompts/vhd_description_zeroshot_time_token_fmt_v2.txt"
# GT_FILE="${ANNO_DIR}/highlight_${SPLIT}_release.jsonl"
# ASR_DIR='data/QVhighlights/whisper_outputs_with_time/tiny.en.cleaned/val/'

NUM_FRAME=96
OUTPUT_DIR=${DIR}/${TASK}_videollama_slot_fm96_fmt
CFG_PATH=""


python evaluate.py --anno_path ${ANNO_DIR} --video_path ${VIDEO_DIR} --gpu_id 0 \
--task ${TASK} --dataset ${DATASET} --output_dir ${OUTPUT_DIR} --split ${SPLIT} --num_frames ${NUM_FRAME} --batch_size 1 \
--prompt_file ${PROMPT_FILE} --timechat_model_path ${MODEL_DIR} --cfg_path yourpath/VTG-LLM/eval_configs/videollama-slot-96-interpolation.yaml

cd metrics/${TASK}
python eval_${TASK}.py --pred_file "${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.json" --gt_file ${GT_FILE} | tee "${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.txt"
cd ../..