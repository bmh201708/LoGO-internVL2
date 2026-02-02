#!/bin/bash
# LOGO Evaluation Script for Qwen2-VL-7B-Instruct
#
# 使用方法:
#   bash logo/evaluate_logo_qwen2vl.sh            # 默认配置
#   bash logo/evaluate_logo_qwen2vl.sh 5          # 指定 GPU
#   bash logo/evaluate_logo_qwen2vl.sh 5 3        # 指定 GPU 和 top_k
#   bash logo/evaluate_logo_qwen2vl.sh 5 3 app    # 仅测试 App-level
#   bash logo/evaluate_logo_qwen2vl.sh 5 3 cat    # 仅测试 Category-level

set -e

# 配置
GPU_ID=${1:-5}
TOP_K=${2:-3}
TEST_TYPE=${3:-"all"}  # all, app, cat

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=100000
export MAX_NUM=12

# 进入项目目录
cd "$(dirname "$0")/.."

echo "==========================================="
echo "LOGO Evaluation - Qwen2-VL-7B-Instruct"
echo "==========================================="
echo "GPU ID: $GPU_ID"
echo "Top-K:  $TOP_K"
echo "Type:   $TEST_TYPE"
echo "==========================================="

# 构建命令
CMD="python logo/evaluate_logo.py --model_type qwen2-vl-7b-instruct --gpu $GPU_ID --top_k $TOP_K --signal_type norm"

case $TEST_TYPE in
    app)
        CMD="$CMD --app_only"
        echo "Mode: App-level only (14 App LoRAs)"
        ;;
    cat|category)
        CMD="$CMD --category_only"
        echo "Mode: Category-level only (5 Category LoRAs)"
        ;;
    val100)
        CMD="$CMD --val100"
        echo "Mode: Val_100.jsonl quick test"
        ;;
    all)
        echo "Mode: Full evaluation (App + Category)"
        ;;
    debug)
        CMD="$CMD --debug"
        echo "Mode: Debug (fewer samples)"
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Valid types: all, app, cat, debug"
        exit 1
        ;;
esac

echo ""
echo "Running: $CMD"
echo "==========================================="
echo ""

eval $CMD

echo ""
echo "==========================================="
echo "Evaluation Complete!"
echo "==========================================="
