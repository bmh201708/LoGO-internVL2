#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=400000
export MAX_NUM=6

# 使用 global_lora_5（最新的 checkpoint）
LORA_CKPT="/home/hmpiao/hmpiao/jinyike/FedMABench/lora_category_internvl2-2b/category_lora_Office_internvl2-2b/internvl2-2b/v5-20260117-122844/global_lora_5"

swift infer \
  --ckpt_dir "$LORA_CKPT" \
  --model_type internvl2-2b \
  --model_id_or_path /home/hmpiao/hmpiao/InternVL2-2B-ModelScope/OpenGVLab/InternVL2-2B \
  --sft_type lora