#!/bin/bash

# 参考 FedMABench 的方式，使用 JSONL 数据集进行推理
# 使用方法: bash infer/infer.sh [DATASET_PATH]
# 默认使用 data/Val_100.jsonl

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=300000
export MAX_NUM=6

# 默认数据集
GPU_ID=${1:-5}
TEST_DATA="${2:-data/Val_100.jsonl}"

# 使用 global_lora_5（最新的 checkpoint）
# 注意：你需要确保这个路径是正确的，或者根据需要修改
LORA_CKPT="/home/hmpiao/hmpiao/jinyike/FedMABench/lora_category_internvl2-2b/category_lora_Entertainment_internvl2-2b/internvl2-2b/v4-20260119-232917/global_lora_2"
OUTPUT_BASE="./output"

echo "Running inference with dataset: $TEST_DATA"
echo "LoRA Checkpoint: $LORA_CKPT"

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "==========================================="
echo "InternVL2 Infer测试"
echo "==========================================="
echo "GPU: $GPU_ID"
echo "LoRA: $LORA_CKPT"
echo "Dataset: $TEST_DATA"
echo "==========================================="

swift infer \
  --ckpt_dir "$LORA_CKPT" \
  --model_type internvl2-2b \
  --model_id_or_path /data0/piaohongming/InternVL2-2B \
  --sft_type lora \
  --val_dataset "$TEST_DATA" \
  --result_dir "$OUTPUT_BASE/val_100"

lora_jsonl=$(find "$OUTPUT_BASE/val_100" -name "*.jsonl" 2>/dev/null | head -n 1)
if [ -n "$lora_jsonl" ]; then
    echo "==========================================="
    echo "开始评测正确率"
    echo "==========================================="
    echo "Results file: $lora_jsonl"
    echo ""
    echo ">>> Accuracy:"
    python evaluation/test_swift.py --data_path "$lora_jsonl" | grep -E "Step-level|Episode"
else
    echo "[WARNING!!!!!] No result file found"
fi

echo ""
echo "==========================================="
echo "Test Completed!"
echo "Results saved in: $OUTPUT_BASE"
echo "==========================================="
