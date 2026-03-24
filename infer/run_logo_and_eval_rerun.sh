#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Runtime config (can be overridden by env vars or CLI args).
# nohup bash infer/run_logo_and_eval_rerun.sh --gpu_id 6 > /data0/piaohongming/jinyike/LoGO-internVL2/rerun_failed_qwen7b.log 2>&1 &
PYTHON_BIN="${PYTHON_BIN:-/data0/piaohongming/envs/LoGO/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MODEL_TYPE="${MODEL_TYPE:-qwen2-vl-7b-instruct}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/data0/piaohongming/jinyike/data/data-test}"
TEST_FILE_SUFFIX="${TEST_FILE_SUFFIX:-_train.jsonl}"
LORA_TYPE="${LORA_TYPE:-app}"
LORA_POOL="${LORA_POOL:-amazon,clock,ebay,etsy,flipkart,google_drive,reminder,youtube}"
MERGE_METHOD="${MERGE_METHOD:-mixture}"
TOP_K="${TOP_K:-3}"
TARGET_BLOCK_IDX="${TARGET_BLOCK_IDX:--1}"
TOKEN_POSITION="${TOKEN_POSITION:-last}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
USE_BASELINE_CALIBRATION="${USE_BASELINE_CALIBRATION:-0}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/output/rerun_failed}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${OUTPUT_BASE}/${RUN_TAG}"

# Failed tasks to rerun: app:signal_type:inference_mode
FAILED_TASKS=(
  "reminder:norm:step"
  "calendar:entropy:step"
  "google_maps:norm:episode"
)

usage() {
  cat <<'EOF'
Usage:
  bash infer/run_logo_and_eval_rerun.sh [options]

Options:
  --python_bin PATH
  --gpu_id ID
  --device_map MAP
  --model_type TYPE
  --test_data_root DIR
  --test_file_suffix SUFFIX
  --lora_type TYPE
  --lora_pool a,b,c
  --merge_method METHOD
  --top_k K
  --target_block_idx IDX
  --token_position POS
  --temperature T
  --seed S
  --num_samples N
  --use_baseline_calibration | --no_baseline_calibration
  --output_base DIR
  --run_tag TAG
EOF
}

CLI_ARGS=("$@")
i=0
while (( i < ${#CLI_ARGS[@]} )); do
  arg="${CLI_ARGS[$i]}"
  arg_lc="${arg,,}"
  case "${arg_lc}" in
    -h|--help)
      usage
      exit 0
      ;;
    --python_bin|--gpu_id|--device_map|--model_type|--test_data_root|--test_file_suffix|--lora_type|--lora_pool|--merge_method|--top_k|--target_block_idx|--token_position|--temperature|--seed|--num_samples|--output_base|--run_tag)
      if (( i + 1 >= ${#CLI_ARGS[@]} )); then
        echo "[ERROR] Missing value for argument: ${arg}"
        exit 2
      fi
      val="${CLI_ARGS[$((i + 1))]}"
      case "${arg_lc}" in
        --python_bin) PYTHON_BIN="${val}" ;;
        --gpu_id) GPU_ID="${val}" ;;
        --device_map) DEVICE_MAP="${val}" ;;
        --model_type) MODEL_TYPE="${val}" ;;
        --test_data_root) TEST_DATA_ROOT="${val}" ;;
        --test_file_suffix) TEST_FILE_SUFFIX="${val}" ;;
        --lora_type) LORA_TYPE="${val}" ;;
        --lora_pool) LORA_POOL="${val}" ;;
        --merge_method) MERGE_METHOD="${val}" ;;
        --top_k) TOP_K="${val}" ;;
        --target_block_idx) TARGET_BLOCK_IDX="${val}" ;;
        --token_position) TOKEN_POSITION="${val}" ;;
        --temperature) TEMPERATURE="${val}" ;;
        --seed) SEED="${val}" ;;
        --num_samples) NUM_SAMPLES="${val}" ;;
        --output_base) OUTPUT_BASE="${val}" ;;
        --run_tag) RUN_TAG="${val}" ;;
      esac
      i=$((i + 2))
      ;;
    --python_bin=*|--gpu_id=*|--device_map=*|--model_type=*|--test_data_root=*|--test_file_suffix=*|--lora_type=*|--lora_pool=*|--merge_method=*|--top_k=*|--target_block_idx=*|--token_position=*|--temperature=*|--seed=*|--num_samples=*|--output_base=*|--run_tag=*)
      val="${arg#*=}"
      case "${arg_lc}" in
        --python_bin=*) PYTHON_BIN="${val}" ;;
        --gpu_id=*) GPU_ID="${val}" ;;
        --device_map=*) DEVICE_MAP="${val}" ;;
        --model_type=*) MODEL_TYPE="${val}" ;;
        --test_data_root=*) TEST_DATA_ROOT="${val}" ;;
        --test_file_suffix=*) TEST_FILE_SUFFIX="${val}" ;;
        --lora_type=*) LORA_TYPE="${val}" ;;
        --lora_pool=*) LORA_POOL="${val}" ;;
        --merge_method=*) MERGE_METHOD="${val}" ;;
        --top_k=*) TOP_K="${val}" ;;
        --target_block_idx=*) TARGET_BLOCK_IDX="${val}" ;;
        --token_position=*) TOKEN_POSITION="${val}" ;;
        --temperature=*) TEMPERATURE="${val}" ;;
        --seed=*) SEED="${val}" ;;
        --num_samples=*) NUM_SAMPLES="${val}" ;;
        --output_base=*) OUTPUT_BASE="${val}" ;;
        --run_tag=*) RUN_TAG="${val}" ;;
      esac
      i=$((i + 1))
      ;;
    --no_baseline_calibration)
      USE_BASELINE_CALIBRATION=0
      i=$((i + 1))
      ;;
    --use_baseline_calibration)
      USE_BASELINE_CALIBRATION=1
      i=$((i + 1))
      ;;
    *)
      echo "[ERROR] Unknown argument: ${arg}"
      usage
      exit 2
      ;;
  esac
done

RUN_DIR="${OUTPUT_BASE}/${RUN_TAG}"
mkdir -p "${RUN_DIR}"

extract_jsonl_from_infer_log() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    echo ""
    return
  fi
  grep -E 'Results saved to:' "${log_file}" \
    | tail -n 1 \
    | sed -E 's/.*Results saved to:[[:space:]]*//'
}

extract_metric_from_eval() {
  local eval_log="$1"
  local mode="$2"
  if [[ "${mode}" == "step" ]]; then
    grep -E 'Step-level Accuracy' "${eval_log}" | head -n 1 | awk -F':' '{gsub(/^ +| +$/,"",$2); print $2}'
  else
    grep -E 'Episode-level Accuracy' "${eval_log}" | head -n 1 | awk -F':' '{gsub(/^ +| +$/,"",$2); print $2}'
  fi
}

SUMMARY_TSV="${RUN_DIR}/rerun_summary.tsv"
SUMMARY_CSV="${RUN_DIR}/rerun_summary.csv"
echo -e "app\tsignal_type\tmode\tmetric\tresult_jsonl\tstatus" > "${SUMMARY_TSV}"
echo "app,signal_type,mode,metric,result_jsonl,status" > "${SUMMARY_CSV}"

echo "==========================================="
echo "Rerun Failed LOGO Tasks"
echo "==========================================="
echo "RUN_DIR      : ${RUN_DIR}"
echo "PYTHON_BIN   : ${PYTHON_BIN}"
echo "GPU_ID       : ${GPU_ID}"
echo "MODEL_TYPE   : ${MODEL_TYPE}"
echo "DEVICE_MAP   : ${DEVICE_MAP}"
echo "MERGE_METHOD : ${MERGE_METHOD}"
echo "TOP_K        : ${TOP_K}"
echo "TGT_BLOCK    : ${TARGET_BLOCK_IDX}"
echo "TOKEN_POS    : ${TOKEN_POSITION}"
echo "TEMP         : ${TEMPERATURE}"
echo "SEED         : ${SEED}"
echo "BASELINE     : ${USE_BASELINE_CALIBRATION}"
echo "LORA_TYPE    : ${LORA_TYPE}"
echo "LORA_POOL    : ${LORA_POOL}"
echo "TEST_ROOT    : ${TEST_DATA_ROOT}"
echo "TASKS        : ${#FAILED_TASKS[@]}"
echo "==========================================="

cd "${REPO_ROOT}"

FAIL_COUNT=0

for task in "${FAILED_TASKS[@]}"; do
  IFS=':' read -r app signal_type mode <<< "${task}"
  test_file="${TEST_DATA_ROOT}/app/${app}${TEST_FILE_SUFFIX}"
  mode_dir="${RUN_DIR}/${app}/${signal_type}/${mode}"
  mkdir -p "${mode_dir}"
  infer_log="${mode_dir}/infer.log"
  eval_log="${mode_dir}/eval.log"

  if [[ ! -f "${test_file}" ]]; then
    echo "[ERROR] Missing test file: ${test_file}"
    echo -e "${app}\t${signal_type}\t${mode}\tN/A\tN/A\tmissing_test_data" >> "${SUMMARY_TSV}"
    echo "${app},${signal_type},${mode},N/A,N/A,missing_test_data" >> "${SUMMARY_CSV}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi

  cmd=(
    "${PYTHON_BIN}" logo/infer_logo.py
    --model_type "${MODEL_TYPE}"
    --gpu_id "${GPU_ID}"
    --device_map "${DEVICE_MAP}"
    --test_data "${test_file}"
    --signal_type "${signal_type}"
    --merge_method "${MERGE_METHOD}"
    --inference_mode "${mode}"
    --lora_type "${LORA_TYPE}"
    --lora_pool "${LORA_POOL}"
    --top_k "${TOP_K}"
    --target_block_idx "${TARGET_BLOCK_IDX}"
    --token_position "${TOKEN_POSITION}"
    --temperature "${TEMPERATURE}"
    --seed "${SEED}"
    --output_dir "${mode_dir}"
  )
  if [[ "${USE_BASELINE_CALIBRATION}" != "1" ]]; then
    cmd+=(--no_baseline_calibration)
  fi
  if [[ -n "${NUM_SAMPLES}" ]]; then
    cmd+=(--num_samples "${NUM_SAMPLES}")
  fi

  echo ""
  echo "-------------------------------------------"
  echo "Rerun task: app=${app}, signal_type=${signal_type}, mode=${mode}"
  echo "test_file=${test_file}"
  echo "output_dir=${mode_dir}"
  echo "-------------------------------------------"

  if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" 2>&1 | tee "${infer_log}"; then
    echo "[ERROR] Inference failed for ${task}"
    echo -e "${app}\t${signal_type}\t${mode}\tN/A\tN/A\tinfer_failed" >> "${SUMMARY_TSV}"
    echo "${app},${signal_type},${mode},N/A,N/A,infer_failed" >> "${SUMMARY_CSV}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi

  result_jsonl="$(extract_jsonl_from_infer_log "${infer_log}")"
  if [[ -z "${result_jsonl}" || ! -f "${result_jsonl}" ]]; then
    echo "[ERROR] Could not locate result JSONL for ${task}"
    echo -e "${app}\t${signal_type}\t${mode}\tN/A\tN/A\tmissing_result_jsonl" >> "${SUMMARY_TSV}"
    echo "${app},${signal_type},${mode},N/A,N/A,missing_result_jsonl" >> "${SUMMARY_CSV}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi

  if ! "${PYTHON_BIN}" evaluation/evaluate_inference.py --results_path "${result_jsonl}" 2>&1 | tee "${eval_log}"; then
    echo "[ERROR] Evaluation failed for ${task}"
    echo -e "${app}\t${signal_type}\t${mode}\tN/A\t${result_jsonl}\teval_failed" >> "${SUMMARY_TSV}"
    echo "${app},${signal_type},${mode},N/A,${result_jsonl},eval_failed" >> "${SUMMARY_CSV}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi

  metric="$(extract_metric_from_eval "${eval_log}" "${mode}")"
  if [[ -z "${metric}" ]]; then
    metric="N/A"
  fi
  echo -e "${app}\t${signal_type}\t${mode}\t${metric}\t${result_jsonl}\tok" >> "${SUMMARY_TSV}"
  echo "${app},${signal_type},${mode},${metric},${result_jsonl},ok" >> "${SUMMARY_CSV}"
done

echo ""
echo "==========================================="
echo "Rerun Summary"
echo "==========================================="
printf "%-14s | %-10s | %-8s | %-16s | %-8s\n" "app" "signal" "mode" "metric" "status"
printf "%-14s-+-%-10s-+-%-8s-+-%-16s-+-%-8s\n" "--------------" "----------" "--------" "----------------" "--------"
tail -n +2 "${SUMMARY_TSV}" | while IFS=$'\t' read -r app signal mode metric result_jsonl status; do
  printf "%-14s | %-10s | %-8s | %-16s | %-8s\n" "${app}" "${signal}" "${mode}" "${metric}" "${status}"
done
echo "==========================================="
echo "Detailed TSV: ${SUMMARY_TSV}"
echo "Detailed CSV: ${SUMMARY_CSV}"
echo "==========================================="

if (( FAIL_COUNT > 0 )); then
  echo "[ERROR] Rerun finished with ${FAIL_COUNT} failed task(s)."
  exit 1
fi
