#!/bin/bash

# LOGO Inference Script for InternVL2
# Usage: bash logo/run_logo.sh [GPU_ID] [TEST_DATA] [TOP_K] [SIGNAL_TYPE]

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=300000
export MAX_NUM=6

# Arguments with defaults
GPU_ID=${1:-0}
TEST_DATA="${2:-data/Val_100.jsonl}"
TOP_K="${3:-5}"
SIGNAL_TYPE="${4:-norm}"
OUTPUT_DIR="${5:-output/logo_results}"

echo "==========================================="
echo "LOGO Inference for InternVL2"
echo "==========================================="
echo "GPU: $GPU_ID"
echo "Test Data: $TEST_DATA"
echo "Top-K: $TOP_K"
echo "Signal Type: $SIGNAL_TYPE"
echo "Output Dir: $OUTPUT_DIR"
echo "==========================================="

export CUDA_VISIBLE_DEVICES=$GPU_ID

python logo/infer_logo.py \
    --test_data "$TEST_DATA" \
    --app_config config/app_loras_config_internvl2.json \
    --category_config config/category_loras_config_internvl2.json \
    --top_k $TOP_K \
    --signal_type $SIGNAL_TYPE \
    --output_dir "$OUTPUT_DIR"

# Find the latest result file and run evaluation
result_file=$(find "$OUTPUT_DIR" -name "logo_results_*.jsonl" 2>/dev/null | sort -r | head -n 1)

if [ -n "$result_file" ]; then
    echo ""
    echo "==========================================="
    echo "Running Evaluation"
    echo "==========================================="
    echo "Results file: $result_file"
    python evaluation/test_swift.py --data_path "$result_file"
else
    echo "[WARNING] No result file found in $OUTPUT_DIR"
fi

echo ""
echo "==========================================="
echo "LOGO Inference Complete!"
echo "==========================================="
