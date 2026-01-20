#!/bin/bash

# Single LoRA Inference Script for InternVL2
# 
# 只加载并使用一个 LoRA 进行推理（用于与 LOGO 多 LoRA 对比）
#
# Usage: 
#   bash logo/run_single_lora.sh [GPU_ID] [LORA_NAME] [TEST_DATA]
#   
# Examples:
#   # 使用 app_lora_youtube
#   bash logo/run_single_lora.sh 5 app_lora_youtube data/Val_100.jsonl
#   
#   # 使用 category_lora_entertainment
#   bash logo/run_single_lora.sh 5 category_lora_entertainment data/Val_100.jsonl
#   
#   # 列出所有可用的 LoRA
#   bash logo/run_single_lora.sh --list
#
# Available LoRAs (14 App + 5 Category):
#   App-level: adidas, amazon, calendar, clock, decathlon, ebay, etsy, 
#              flipkart, google_drive, google_maps, gmail, pinterest, 
#              ticktick, youtube
#   Category-level: shopping, entertainment, lives, office, travel

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=300000
export MAX_NUM=6

# 配置文件路径
APP_CONFIG="config/app_loras_config_internvl2.json"
CATEGORY_CONFIG="config/category_loras_config_internvl2.json"
MODEL_PATH="/home/hmpiao/hmpiao/InternVL2-2B-ModelScope/OpenGVLab/InternVL2-2B"

# 从配置文件获取 LoRA 路径的函数
get_lora_path() {
    local lora_name="$1"
    local path=""
    
    # 先在 app config 中查找
    if [ -f "$APP_CONFIG" ]; then
        path=$(python3 -c "
import json
with open('$APP_CONFIG') as f:
    configs = json.load(f)
for c in configs:
    if c['lora_name'] == '$lora_name':
        print(c['lora_path'])
        break
" 2>/dev/null)
    fi
    
    # 如果没找到，在 category config 中查找
    if [ -z "$path" ] && [ -f "$CATEGORY_CONFIG" ]; then
        path=$(python3 -c "
import json
with open('$CATEGORY_CONFIG') as f:
    configs = json.load(f)
for c in configs:
    if c['lora_name'] == '$lora_name':
        print(c['lora_path'])
        break
" 2>/dev/null)
    fi
    
    echo "$path"
}

# 列出所有可用的 LoRA
list_loras() {
    echo "==========================================="
    echo "Available LoRAs"
    echo "==========================================="
    echo ""
    echo "App-level LoRAs (from $APP_CONFIG):"
    if [ -f "$APP_CONFIG" ]; then
        python3 -c "
import json
with open('$APP_CONFIG') as f:
    configs = json.load(f)
for c in configs:
    print(f\"  - {c['lora_name']}\")
"
    else
        echo "  [Config file not found]"
    fi
    echo ""
    echo "Category-level LoRAs (from $CATEGORY_CONFIG):"
    if [ -f "$CATEGORY_CONFIG" ]; then
        python3 -c "
import json
with open('$CATEGORY_CONFIG') as f:
    configs = json.load(f)
for c in configs:
    print(f\"  - {c['lora_name']}\")
"
    else
        echo "  [Config file not found]"
    fi
    echo ""
    echo "Usage: bash logo/run_single_lora.sh [GPU_ID] [LORA_NAME] [TEST_DATA]"
    echo "Example: bash logo/run_single_lora.sh 5 app_lora_youtube data/Val_100.jsonl"
}

# 检查 --list 参数
if [ "$1" == "--list" ] || [ "$1" == "-l" ]; then
    list_loras
    exit 0
fi

# Arguments with defaults
GPU_ID=${1:-5}
LORA_NAME="${2:-app_lora_youtube}"
TEST_DATA="${3:-data/Val_100.jsonl}"
OUTPUT_DIR="${4:-output/single_lora_results}"

# 获取 LoRA 路径
LORA_PATH=$(get_lora_path "$LORA_NAME")

if [ -z "$LORA_PATH" ]; then
    echo "[ERROR] Unknown LoRA name: $LORA_NAME"
    echo ""
    list_loras
    exit 1
fi

# 检查 LoRA 路径是否存在
if [ ! -d "$LORA_PATH" ]; then
    echo "[ERROR] LoRA path not found: $LORA_PATH"
    exit 1
fi

# 检查测试数据是否存在
if [ ! -f "$TEST_DATA" ]; then
    echo "[ERROR] Test data not found: $TEST_DATA"
    exit 1
fi

echo "==========================================="
echo "Single LoRA Inference for InternVL2"
echo "==========================================="
echo "GPU ID:        $GPU_ID"
echo "LoRA Name:     $LORA_NAME"
echo "LoRA Path:     $LORA_PATH"
echo "Test Data:     $TEST_DATA"
echo "Output Dir:    $OUTPUT_DIR"
echo "==========================================="

export CUDA_VISIBLE_DEVICES=$GPU_ID

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULT_SUBDIR="$OUTPUT_DIR/${LORA_NAME}_${TIMESTAMP}"
mkdir -p "$RESULT_SUBDIR"

echo ""
echo "Starting inference..."
echo ""

# 使用 swift infer 进行推理
swift infer \
    --ckpt_dir "$LORA_PATH" \
    --model_type internvl2-2b \
    --model_id_or_path "$MODEL_PATH" \
    --sft_type lora \
    --val_dataset "$TEST_DATA" \
    --result_dir "$RESULT_SUBDIR"

# 查找结果文件
result_file=$(find "$RESULT_SUBDIR" -name "*.jsonl" 2>/dev/null | head -n 1)

if [ -n "$result_file" ]; then
    echo ""
    echo "==========================================="
    echo "Running Evaluation"
    echo "==========================================="
    echo "Results file: $result_file"
    echo ""
    
    if [ -f "evaluation/test_swift.py" ]; then
        echo ">>> Accuracy for $LORA_NAME:"
        python evaluation/test_swift.py --data_path "$result_file" | grep -E "Step-level|Episode"
    else
        echo "[INFO] Evaluation script not found, skipping evaluation"
        echo "[INFO] Results saved to: $result_file"
    fi
else
    echo "[WARNING] No result file found in $RESULT_SUBDIR"
fi

echo ""
echo "==========================================="
echo "Single LoRA Inference Complete!"
echo "==========================================="
echo "LoRA: $LORA_NAME"
echo "Results: $RESULT_SUBDIR"
echo "==========================================="
