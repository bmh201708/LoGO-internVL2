#!/bin/bash

# LOGO Inference Script for InternVL2
# 
# Based on: "LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging"
# 
# Usage: 
#   bash logo/run_logo.sh [GPU_ID] [TEST_DATA] [TOP_K] [SIGNAL_TYPE] [MERGE_METHOD] [NO_BASELINE] [LORA_TYPE] [DEBUG_SIGNALS] [OUTPUT_DIR]
#   
# Examples:
#   # Default: use norm signal, top-3, mixture mode, no baseline calibration
#   bash logo/run_logo.sh 5 data/Val_100.jsonl 3 norm mixture
#   
#   # Only use app LoRAs (exclude category LoRAs)
#   bash logo/run_logo.sh 5 data/Val_100.jsonl 3 norm mixture true app
#   
#   # Only use category LoRAs (exclude app LoRAs)
#   bash logo/run_logo.sh 5 data/Val_100.jsonl 3 norm mixture true category
#   
#   # Debug mode: print all raw signals
#   bash logo/run_logo.sh 5 data/Val_100.jsonl 3 norm mixture true all true
#   
#   # Use uniform weights (baseline)
#   bash logo/run_logo.sh 5 data/Val_100.jsonl 3 uniform mixture
#
# Signal Types:
#   - norm: L2 norm of LoRA projection outputs (Equation 2 in paper)
#   - entropy: Inverse entropy of LoRA projection outputs (Equation 3 in paper)
#   - embedding: Cosine similarity with precomputed embeddings
#   - uniform: Equal weights for all adapters (baseline)
#
# Merge Methods:
#   - mixture: Output-level weighted sum (论文 Section 3.3, 推荐)
#             output = Σ(wᵢ × LoRAᵢ(x))
#   - add_weighted_adapter: Parameter-level merging using PEFT's add_weighted_adapter

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=300000
export MAX_NUM=6

# Arguments with defaults
GPU_ID=${1:-5}
TEST_DATA="${2:-data/Val_100.jsonl}"
TOP_K="${3:-3}"
SIGNAL_TYPE="${4:-norm}"
MERGE_METHOD="${5:-mixture}"  # mixture (output-level) or add_weighted_adapter (parameter-level)
NO_BASELINE="${6:-true}"     # true to disable baseline calibration
LORA_TYPE="${7:-all}"        # all, app, or category - which LoRA types to load
DEBUG_SIGNALS="${8:-false}"  # true to print all raw signals for debugging
OUTPUT_DIR="${9:-output/logo_results}"
COMBINATION_TYPE="${10:-linear}"  # for add_weighted_adapter: linear, svd, cat

# Validate merge method
if [ "$MERGE_METHOD" != "mixture" ] && [ "$MERGE_METHOD" != "add_weighted_adapter" ]; then
    echo "[ERROR] Invalid merge method: $MERGE_METHOD"
    echo "        Valid options: mixture, add_weighted_adapter"
    exit 1
fi

echo "==========================================="
echo "LOGO Inference for InternVL2"
echo "==========================================="
echo "GPU ID:           $GPU_ID"
echo "Test Data:        $TEST_DATA"
echo "Top-K:            $TOP_K"
echo "Signal Type:      $SIGNAL_TYPE"
echo "Merge Method:     $MERGE_METHOD"
echo "No Baseline:      $NO_BASELINE"
echo "LoRA Type:        $LORA_TYPE"
echo "Debug Signals:    $DEBUG_SIGNALS"
if [ "$MERGE_METHOD" == "add_weighted_adapter" ]; then
echo "Combination Type: $COMBINATION_TYPE"
fi
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

# Build command
CMD="python logo/infer_logo.py \
    --test_data $TEST_DATA \
    --app_config config/app_loras_config_internvl2.json \
    --category_config config/category_loras_config_internvl2.json \
    --top_k $TOP_K \
    --signal_type $SIGNAL_TYPE \
    --merge_method $MERGE_METHOD \
    --combination_type $COMBINATION_TYPE \
    --lora_type $LORA_TYPE \
    --output_dir $OUTPUT_DIR \
    --target_block_idx -1 \
    --token_position last"

# Add no_baseline_calibration flag if requested
if [ "$NO_BASELINE" == "true" ]; then
    CMD="$CMD --no_baseline_calibration"
fi

# Add debug_signals flag if requested
if [ "$DEBUG_SIGNALS" == "true" ]; then
    CMD="$CMD --debug_signals"
fi

eval $CMD

# Find the latest result file and run evaluation (by modification time, not alphabetically)
result_file=$(ls -t "$OUTPUT_DIR"/logo_results_*.jsonl 2>/dev/null | head -n 1)

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
