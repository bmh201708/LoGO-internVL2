#!/bin/bash

# LOGO Inference Script for InternVL2
# 
# Based on: "LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging"
# 
# Usage: 
#   bash logo/run_logo.sh [GPU_ID] [TEST_DATA] [TOP_K] [SIGNAL_TYPE] [OUTPUT_DIR]
#   
# Examples:
#   # Default: use norm signal, top-5
#   bash logo/run_logo.sh 5 data/Val_100.jsonl 5 norm
#   
#   # Use entropy signal
#   bash logo/run_logo.sh 5 data/Val_100.jsonl 5 entropy
#   
#   # Use uniform weights (baseline)
#   bash logo/run_logo.sh 5 data/Val_100.jsonl 5 uniform
#
# Signal Types:
#   - norm: L2 norm of LoRA projection outputs (Equation 2 in paper)
#   - entropy: Inverse entropy of LoRA projection outputs (Equation 3 in paper)
#   - embedding: Cosine similarity with precomputed embeddings
#   - uniform: Equal weights for all adapters (baseline)

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=300000
export MAX_NUM=6

# Arguments with defaults
GPU_ID=${1:-5}
TEST_DATA="${2:-data/Val_100.jsonl}"
TOP_K="${3:-5}"
SIGNAL_TYPE="${4:-norm}"
OUTPUT_DIR="${5:-output/logo_results}"
COMBINATION_TYPE="${6:-linear}"

echo "==========================================="
echo "LOGO Inference for InternVL2"
echo "==========================================="
echo "GPU ID:           $GPU_ID"
echo "Test Data:        $TEST_DATA"
echo "Top-K:            $TOP_K"
echo "Signal Type:      $SIGNAL_TYPE"
echo "Combination Type: $COMBINATION_TYPE"
echo "Output Dir:       $OUTPUT_DIR"
echo "==========================================="

# Check if test data exists
if [ ! -f "$TEST_DATA" ]; then
    echo "[ERROR] Test data not found: $TEST_DATA"
    exit 1
fi

# Check if config files exist
if [ ! -f "config/app_loras_config_internvl2.json" ] && [ ! -f "config/category_loras_config_internvl2.json" ]; then
    echo "[WARNING] No LoRA config files found in config/"
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo ""
echo "Starting inference..."
echo ""

python logo/infer_logo.py \
    --test_data "$TEST_DATA" \
    --app_config config/app_loras_config_internvl2.json \
    --category_config config/category_loras_config_internvl2.json \
    --top_k $TOP_K \
    --signal_type $SIGNAL_TYPE \
    --combination_type $COMBINATION_TYPE \
    --output_dir "$OUTPUT_DIR" \
    --target_block_idx -1 \
    --token_position last

# Find the latest result file and run evaluation
result_file=$(find "$OUTPUT_DIR" -name "logo_results_*.jsonl" 2>/dev/null | sort -r | head -n 1)

if [ -n "$result_file" ]; then
    echo ""
    echo "==========================================="
    echo "Running Evaluation"
    echo "==========================================="
    echo "Results file: $result_file"
    
    if [ -f "evaluation/test_swift.py" ]; then
        python evaluation/test_swift.py --data_path "$result_file"
    else
        echo "[INFO] Evaluation script not found, skipping evaluation"
        echo "[INFO] Results saved to: $result_file"
    fi
else
    echo "[WARNING] No result file found in $OUTPUT_DIR"
fi

echo ""
echo "==========================================="
echo "LOGO Inference Complete!"
echo "==========================================="
