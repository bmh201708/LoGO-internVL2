#!/bin/bash
# ==============================================================================
# LOGO Evaluation Script
# 
# 对 14 个 App LoRA 测试集和 5 个 Category LoRA 测试集进行完整测评
#
# 测评方法：
# 1. App-level 测试：只加载 14 个 app LoRA 作为候选
# 2. Category-level 测试：只加载 5 个 category LoRA 作为候选
# 3. 使用 LOGO 方法进行动态 LoRA 选择和合并
# 4. 使用 evaluation/test_swift.py 进行评估
#
# 使用方法:
#   bash logo/evaluate_logo.sh [GPU_ID] [TOP_K] [SIGNAL_TYPE]
#
# 参数:
#   GPU_ID      - GPU ID (默认: 5)
#   TOP_K       - 选择的 top-k LoRA 数量 (默认: 3)
#   SIGNAL_TYPE - 信号类型: norm/entropy/uniform (默认: norm)
#
# 示例:
#   bash logo/evaluate_logo.sh 5 3 norm
#   bash logo/evaluate_logo.sh 5 5 entropy
# ==============================================================================

set -e

# 参数解析
GPU_ID=${1:-5}
TOP_K=${2:-3}
SIGNAL_TYPE=${3:-norm}

# 项目路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 配置路径
APP_CONFIG="$PROJECT_ROOT/config/app_loras_config_internvl2.json"
CATEGORY_CONFIG="$PROJECT_ROOT/config/category_loras_config_internvl2.json"

# 测试数据路径
APP_DATA_DIR="$PROJECT_ROOT/data/test_data_by_app"
CATEGORY_DATA_DIR="$PROJECT_ROOT/data/test_data_by_category"

# 输出目录
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/output/logo_evaluation_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

# 日志文件
LOG_FILE="$OUTPUT_DIR/evaluation.log"
SUMMARY_FILE="$OUTPUT_DIR/summary.md"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

# 打印横线
print_divider() {
    log "========================================================================"
}

# 检查文件存在
check_file() {
    if [ ! -f "$1" ]; then
        log_error "File not found: $1"
        exit 1
    fi
}

# 运行推理
run_inference() {
    local test_data=$1
    local lora_config=$2
    local output_file=$3
    local dataset_name=$4
    
    log_info "Running inference on $dataset_name..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_ROOT/logo/infer_logo.py" \
        --test_data "$test_data" \
        --app_config "$lora_config" \
        --category_config "/dev/null" \
        --output_dir "$(dirname "$output_file")" \
        --top_k $TOP_K \
        --signal_type $SIGNAL_TYPE \
        --target_block_idx -1 \
        --token_position last \
        --combination_type linear \
        --max_new_tokens 512 \
        --temperature 0.0 \
        2>&1 | tee -a "$LOG_FILE"
    
    # 获取最新生成的结果文件
    local latest_result=$(ls -t "$(dirname "$output_file")"/logo_results_*.jsonl 2>/dev/null | head -1)
    if [ -n "$latest_result" ] && [ -f "$latest_result" ]; then
        mv "$latest_result" "$output_file"
        log_success "Inference completed: $output_file"
    else
        log_error "Inference output not found"
        return 1
    fi
}

# 运行推理 (Category 模式 - 只使用 category config)
run_inference_category() {
    local test_data=$1
    local lora_config=$2
    local output_file=$3
    local dataset_name=$4
    
    log_info "Running inference on $dataset_name (category mode)..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_ROOT/logo/infer_logo.py" \
        --test_data "$test_data" \
        --app_config "/dev/null" \
        --category_config "$lora_config" \
        --output_dir "$(dirname "$output_file")" \
        --top_k $TOP_K \
        --signal_type $SIGNAL_TYPE \
        --target_block_idx -1 \
        --token_position last \
        --combination_type linear \
        --max_new_tokens 512 \
        --temperature 0.0 \
        2>&1 | tee -a "$LOG_FILE"
    
    # 获取最新生成的结果文件
    local latest_result=$(ls -t "$(dirname "$output_file")"/logo_results_*.jsonl 2>/dev/null | head -1)
    if [ -n "$latest_result" ] && [ -f "$latest_result" ]; then
        mv "$latest_result" "$output_file"
        log_success "Inference completed: $output_file"
    else
        log_error "Inference output not found"
        return 1
    fi
}

# 运行评估
run_evaluation() {
    local result_file=$1
    local dataset_name=$2
    
    log_info "Evaluating $dataset_name..."
    
    local eval_output=$(python "$PROJECT_ROOT/evaluation/test_swift.py" --data_path "$result_file" 2>&1)
    
    echo "$eval_output" | tee -a "$LOG_FILE"
    
    # 提取准确率
    local step_acc=$(echo "$eval_output" | grep -E "Step-level" | grep -oE "[0-9]+\.[0-9]+%?" | head -1)
    local episode_acc=$(echo "$eval_output" | grep -E "Episode-level" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    
    echo "$dataset_name|$step_acc|$episode_acc"
}

# ==============================================================================
# 主程序
# ==============================================================================

print_divider
log "LOGO Evaluation Script"
print_divider
log ""
log "Configuration:"
log "  GPU ID:       $GPU_ID"
log "  Top-K:        $TOP_K"
log "  Signal Type:  $SIGNAL_TYPE"
log "  Output Dir:   $OUTPUT_DIR"
log ""
print_divider

# 检查配置文件
check_file "$APP_CONFIG"
check_file "$CATEGORY_CONFIG"

# 初始化结果汇总
declare -A APP_RESULTS
declare -A CATEGORY_RESULTS

# ==============================================================================
# Part 1: App-level 评测 (使用 14 个 App LoRA)
# ==============================================================================

print_divider
log "Part 1: App-level Evaluation (14 App LoRAs)"
print_divider

APP_OUTPUT_DIR="$OUTPUT_DIR/app_results"
mkdir -p "$APP_OUTPUT_DIR"

APP_TEST_FILES=(
    "adidas_train.jsonl"
    "amazon_train.jsonl"
    "calendar_train.jsonl"
    "clock_train.jsonl"
    "decathlon_train.jsonl"
    "ebay_train.jsonl"
    "etsy_train.jsonl"
    "flipkart_train.jsonl"
    "gmail_train.jsonl"
    "google_drive_train.jsonl"
    "google_maps_train.jsonl"
    "kitchen_stories_train.jsonl"
    "reminder_train.jsonl"
    "youtube_train.jsonl"
)

for test_file in "${APP_TEST_FILES[@]}"; do
    test_path="$APP_DATA_DIR/$test_file"
    
    if [ ! -f "$test_path" ]; then
        log_warning "Test file not found: $test_path"
        continue
    fi
    
    dataset_name=$(basename "$test_file" _train.jsonl)
    output_file="$APP_OUTPUT_DIR/${dataset_name}_results.jsonl"
    
    log ""
    log_info "Processing app: $dataset_name"
    
    # 运行推理
    if run_inference "$test_path" "$APP_CONFIG" "$output_file" "$dataset_name"; then
        # 运行评估
        eval_result=$(run_evaluation "$output_file" "$dataset_name")
        APP_RESULTS["$dataset_name"]="$eval_result"
    else
        APP_RESULTS["$dataset_name"]="$dataset_name|ERROR|ERROR"
    fi
done

# ==============================================================================
# Part 2: Category-level 评测 (使用 5 个 Category LoRA)
# ==============================================================================

print_divider
log "Part 2: Category-level Evaluation (5 Category LoRAs)"
print_divider

CATEGORY_OUTPUT_DIR="$OUTPUT_DIR/category_results"
mkdir -p "$CATEGORY_OUTPUT_DIR"

CATEGORY_TEST_FILES=(
    "entertainment_train.jsonl"
    "lives_train.jsonl"
    "office_train.jsonl"
    "shopping_train.jsonl"
    "traveling_train.jsonl"
)

for test_file in "${CATEGORY_TEST_FILES[@]}"; do
    test_path="$CATEGORY_DATA_DIR/$test_file"
    
    if [ ! -f "$test_path" ]; then
        log_warning "Test file not found: $test_path"
        continue
    fi
    
    dataset_name=$(basename "$test_file" _train.jsonl)
    output_file="$CATEGORY_OUTPUT_DIR/${dataset_name}_results.jsonl"
    
    log ""
    log_info "Processing category: $dataset_name"
    
    # 运行推理 (Category 模式)
    if run_inference_category "$test_path" "$CATEGORY_CONFIG" "$output_file" "$dataset_name"; then
        # 运行评估
        eval_result=$(run_evaluation "$output_file" "$dataset_name")
        CATEGORY_RESULTS["$dataset_name"]="$eval_result"
    else
        CATEGORY_RESULTS["$dataset_name"]="$dataset_name|ERROR|ERROR"
    fi
done

# ==============================================================================
# 生成汇总报告
# ==============================================================================

print_divider
log "Generating Summary Report"
print_divider

cat > "$SUMMARY_FILE" << EOF
# LOGO Evaluation Summary

**Date:** $(date '+%Y-%m-%d %H:%M:%S')
**Configuration:**
- GPU ID: $GPU_ID
- Top-K: $TOP_K
- Signal Type: $SIGNAL_TYPE

---

## App-level Results (14 App LoRAs)

| App | Step-level Accuracy | Episode-level Accuracy |
|-----|---------------------|------------------------|
EOF

for dataset in "${!APP_RESULTS[@]}"; do
    result="${APP_RESULTS[$dataset]}"
    IFS='|' read -r name step episode <<< "$result"
    echo "| $name | $step | $episode |" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" << EOF

---

## Category-level Results (5 Category LoRAs)

| Category | Step-level Accuracy | Episode-level Accuracy |
|----------|---------------------|------------------------|
EOF

for dataset in "${!CATEGORY_RESULTS[@]}"; do
    result="${CATEGORY_RESULTS[$dataset]}"
    IFS='|' read -r name step episode <<< "$result"
    echo "| $name | $step | $episode |" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" << EOF

---

## Output Files

- Log: \`$LOG_FILE\`
- App Results: \`$APP_OUTPUT_DIR/\`
- Category Results: \`$CATEGORY_OUTPUT_DIR/\`

EOF

# ==============================================================================
# 打印最终汇总
# ==============================================================================

print_divider
log ""
log "${GREEN}Evaluation Complete!${NC}"
log ""
log "Results Summary:"
log ""
log "App-level (14 LoRAs):"
for dataset in "${!APP_RESULTS[@]}"; do
    result="${APP_RESULTS[$dataset]}"
    IFS='|' read -r name step episode <<< "$result"
    log "  $name: Step=$step, Episode=$episode"
done
log ""
log "Category-level (5 LoRAs):"
for dataset in "${!CATEGORY_RESULTS[@]}"; do
    result="${CATEGORY_RESULTS[$dataset]}"
    IFS='|' read -r name step episode <<< "$result"
    log "  $name: Step=$step, Episode=$episode"
done
log ""
log "Full report: $SUMMARY_FILE"
print_divider
