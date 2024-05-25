#!/bin/bash

DIR="VTG-LLM"
MODEL_DIR="VTG-LLM/vtgllm/ckpt/ours/slot_videollama_fm96_fmt/20240521002/checkpoint_9.pth"


# TASK='dvc'
# ANNO_DIR='data/VTG-IT/dense_video_caption/Youcook2'
# VIDEO_DIR='data/youcook2/YouCook2_asr_denseCap/youcook2_6fps_224'
# DATASET='youcook'
# SPLIT='val'
# PROMPT_FILE="prompts/${TASK}.txt"
# GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"


# TASK='tvg'
# ANNO_DIR="data/VTG-IT/moment_retrieval/Charades"
# VIDEO_DIR="data/Charades/videos"
# DATASET='charades'
# SPLIT='test'
# PROMPT_FILE="prompts/mr.txt"
# GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"

# TASK='vhd'
# ANNO_DIR='data/VTG-IT/video_highlight_detection/QVHighlights'
# VIDEO_DIR='data/qvhighlights/videos/val'
# DATASET='qvhighlights'
# SPLIT='val'
# PROMPT_FILE="prompts/vhd.txt"
# GT_FILE="${ANNO_DIR}/highlight_${SPLIT}_release.jsonl"

NUM_FRAME=96
OUTPUT_DIR=${DIR}/${TASK}_videollama_slot_fm96
CFG_PATH=""


ASCEND_RT_VISIBLE_DEVICES=0 python evaluate.py --anno_path ${ANNO_DIR} --video_path ${VIDEO_DIR} --gpu_id 0 \
--task ${TASK} --dataset ${DATASET} --output_dir ${OUTPUT_DIR} --split ${SPLIT} --num_frames ${NUM_FRAME} --batch_size 1 \
--prompt_file ${PROMPT_FILE} --vtgllm_model_path ${MODEL_DIR} --cfg_path eval_configs/videollama-slot-96-interpolation.yaml

cd metrics/${TASK}
python eval_${TASK}.py --pred_file "${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.json" --gt_file ${GT_FILE} | tee "${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.txt"
cd ../..