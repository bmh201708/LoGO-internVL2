#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# nohup bash infer/batch_eval_logo_ood_part2.sh --gpu_id 7 > logo_ood_2.log 2>&1 &

# ===== Runtime config =====
PYTHON_BIN="${PYTHON_BIN:-/data0/piaohongming/envs/LoGO/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MODEL_TYPE="${MODEL_TYPE:-qwen2-vl-7b-instruct}"
TOP_K="${TOP_K:-3}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
TARGET_BLOCK_IDX="${TARGET_BLOCK_IDX:--1}"
TOKEN_POSITION="${TOKEN_POSITION:-last}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
USE_BASELINE_CALIBRATION="${USE_BASELINE_CALIBRATION:-0}"
MERGE_METHOD="${MERGE_METHOD:-mixture}"

# Always evaluate both norm/entropy signal types by default.
# Can be overridden by editing this array.
SIGNAL_TYPES=(
  norm
  entropy
)

# Test dataset root. Expected file pattern: ${TEST_DATA_ROOT}/app/<app>_train.jsonl
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/data0/piaohongming/jinyike/data/data-test}"
TEST_FILE_SUFFIX="${TEST_FILE_SUFFIX:-_train.jsonl}"

# Output root
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/output/batch_eval_logo}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${OUTPUT_BASE}/${RUN_TAG}"

# ===== Candidate pool for LoRA retrieval/merging (APP) =====
# Edit this set to control which APP LoRAs are allowed to be selected/merged.
APP=(
  amazon clock ebay etsy flipkart google_drive reminder youtube
)

# ===== Test set apps (TEST_APP) =====
# Edit this set to control which app test files are evaluated.
# TEST_APP can be different from APP (IID/OOD evaluation).
TEST_APP=(
  gmail google_maps kitchen_stories
)

usage() {
  cat <<'EOF'
Usage:
  bash infer/batch_eval_logo.sh [options]

Options:
  --python_bin PATH
  --gpu_id ID
  --device_map MAP
  --model_type TYPE
  --top_k K
  --num_samples N
  --target_block_idx IDX
  --token_position POS
  --temperature T
  --seed S
  --use_baseline_calibration | --no_baseline_calibration
  --merge_method mixture
  --signal_types a,b       (or --signal_type a)
  --test_data_root DIR
  --test_file_suffix SUF
  --output_base DIR
  --run_tag TAG
  --app_pool a,b,c         (candidate APP LoRA pool)
  --test_app a,b,c         (apps to evaluate)
EOF
}

split_csv_into_array() {
  local csv="$1"
  local -n out_arr="$2"
  local IFS=','
  local raw=()
  local cleaned=()
  local item=""
  read -r -a raw <<< "${csv}"
  for item in "${raw[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    if [[ -n "${item}" ]]; then
      cleaned+=("${item}")
    fi
  done
  out_arr=("${cleaned[@]}")
}

# CLI overrides (preferred over env vars).
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
    --python_bin|--gpu_id|--device_map|--model_type|--top_k|--num_samples|--target_block_idx|--token_position|--temperature|--seed|--merge_method|--signal_types|--signal_type|--test_data_root|--test_file_suffix|--output_base|--run_tag|--app_pool|--test_app)
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
        --top_k) TOP_K="${val}" ;;
        --num_samples) NUM_SAMPLES="${val}" ;;
        --target_block_idx) TARGET_BLOCK_IDX="${val}" ;;
        --token_position) TOKEN_POSITION="${val}" ;;
        --temperature) TEMPERATURE="${val}" ;;
        --seed) SEED="${val}" ;;
        --merge_method) MERGE_METHOD="${val}" ;;
        --signal_types) split_csv_into_array "${val}" SIGNAL_TYPES ;;
        --signal_type) SIGNAL_TYPES=("${val}") ;;
        --test_data_root) TEST_DATA_ROOT="${val}" ;;
        --test_file_suffix) TEST_FILE_SUFFIX="${val}" ;;
        --output_base) OUTPUT_BASE="${val}" ;;
        --run_tag) RUN_TAG="${val}" ;;
        --app_pool) split_csv_into_array "${val}" APP ;;
        --test_app) split_csv_into_array "${val}" TEST_APP ;;
      esac
      i=$((i + 2))
      ;;
    --python_bin=*|--gpu_id=*|--device_map=*|--model_type=*|--top_k=*|--num_samples=*|--target_block_idx=*|--token_position=*|--temperature=*|--seed=*|--merge_method=*|--signal_types=*|--signal_type=*|--test_data_root=*|--test_file_suffix=*|--output_base=*|--run_tag=*|--app_pool=*|--test_app=*)
      val="${arg#*=}"
      case "${arg_lc}" in
        --python_bin=*) PYTHON_BIN="${val}" ;;
        --gpu_id=*) GPU_ID="${val}" ;;
        --device_map=*) DEVICE_MAP="${val}" ;;
        --model_type=*) MODEL_TYPE="${val}" ;;
        --top_k=*) TOP_K="${val}" ;;
        --num_samples=*) NUM_SAMPLES="${val}" ;;
        --target_block_idx=*) TARGET_BLOCK_IDX="${val}" ;;
        --token_position=*) TOKEN_POSITION="${val}" ;;
        --temperature=*) TEMPERATURE="${val}" ;;
        --seed=*) SEED="${val}" ;;
        --merge_method=*) MERGE_METHOD="${val}" ;;
        --signal_types=*) split_csv_into_array "${val}" SIGNAL_TYPES ;;
        --signal_type=*) SIGNAL_TYPES=("${val}") ;;
        --test_data_root=*) TEST_DATA_ROOT="${val}" ;;
        --test_file_suffix=*) TEST_FILE_SUFFIX="${val}" ;;
        --output_base=*) OUTPUT_BASE="${val}" ;;
        --run_tag=*) RUN_TAG="${val}" ;;
        --app_pool=*) split_csv_into_array "${val}" APP ;;
        --test_app=*) split_csv_into_array "${val}" TEST_APP ;;
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

if [[ "${MERGE_METHOD}" != "mixture" ]]; then
  echo "[ERROR] batch_eval_logo.sh currently only supports --merge_method mixture"
  exit 2
fi
if [[ ${#SIGNAL_TYPES[@]} -eq 0 ]]; then
  echo "[ERROR] SIGNAL_TYPES is empty. Use --signal_type or --signal_types."
  exit 2
fi
if [[ ${#APP[@]} -eq 0 ]]; then
  echo "[ERROR] APP candidate pool is empty. Use --app_pool a,b,c."
  exit 2
fi
if [[ ${#TEST_APP[@]} -eq 0 ]]; then
  echo "[ERROR] TEST_APP is empty. Use --test_app a,b,c."
  exit 2
fi

join_by_comma() {
  local IFS=","
  echo "$*"
}

extract_metric_from_eval() {
  local eval_log="$1"
  local keyword="$2"
  local line
  line="$(grep -E "${keyword}" "${eval_log}" | head -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo "N/A"
    return
  fi
  echo "${line}" | awk -F':' '{gsub(/^ +| +$/,"",$2); print $2}'
}

find_latest_jsonl() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo ""
    return
  fi
  find "${dir}" -maxdepth 1 -type f -name 'retriever_results_*.jsonl' -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

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

APP_POOL="$(join_by_comma "${APP[@]}")"
FAIL_COUNT=0

mkdir -p "${RUN_DIR}"
SUMMARY_TSV="${RUN_DIR}/summary.tsv"
SUMMARY_CSV="${RUN_DIR}/summary.csv"
echo -e "app\tsignal_type\tstep_accuracy\tepisode_accuracy\tstep_jsonl\tepisode_jsonl" > "${SUMMARY_TSV}"
echo "app,signal_type,step_accuracy,episode_accuracy,step_jsonl,episode_jsonl" > "${SUMMARY_CSV}"

echo "==========================================="
echo "Batch LOGO Evaluation"
echo "==========================================="
echo "RUN_DIR      : ${RUN_DIR}"
echo "MODEL_TYPE   : ${MODEL_TYPE}"
echo "GPU_ID       : ${GPU_ID}"
echo "DEVICE_MAP   : ${DEVICE_MAP}"
echo "MERGE_METHOD : ${MERGE_METHOD}"
echo "SIGNAL_TYPES : $(join_by_comma "${SIGNAL_TYPES[@]}")"
echo "TOP_K        : ${TOP_K}"
echo "TGT_BLOCK    : ${TARGET_BLOCK_IDX}"
echo "TOKEN_POS    : ${TOKEN_POSITION}"
echo "TEMP         : ${TEMPERATURE}"
echo "SEED         : ${SEED}"
echo "BASELINE     : ${USE_BASELINE_CALIBRATION}"
echo "TEST_DATA    : ${TEST_DATA_ROOT}/app/*${TEST_FILE_SUFFIX}"
echo "APP_POOL     : ${APP_POOL}"
echo "TEST_APP     : $(join_by_comma "${TEST_APP[@]}")"
echo "==========================================="

cd "${REPO_ROOT}"

for app in "${TEST_APP[@]}"; do
  test_file="${TEST_DATA_ROOT}/app/${app}${TEST_FILE_SUFFIX}"
  if [[ ! -f "${test_file}" ]]; then
    echo "[WARNING] Skip ${app}: test file not found: ${test_file}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    for signal_type in "${SIGNAL_TYPES[@]}"; do
      echo -e "${app}\t${signal_type}\tN/A\tN/A\tN/A\tN/A" >> "${SUMMARY_TSV}"
      echo "${app},${signal_type},N/A,N/A,N/A,N/A" >> "${SUMMARY_CSV}"
    done
    continue
  fi

  for signal_type in "${SIGNAL_TYPES[@]}"; do
    echo ""
    echo "-------------------------------------------"
    echo "Evaluating app=${app}, signal_type=${signal_type}, merge_method=${MERGE_METHOD}"
    echo "test_file=${test_file}"
    echo "-------------------------------------------"

    step_acc="N/A"
    episode_acc="N/A"
    step_jsonl="N/A"
    episode_jsonl="N/A"

    for mode in step episode; do
      mode_dir="${RUN_DIR}/${app}/${signal_type}/${mode}"
      mkdir -p "${mode_dir}"
      infer_log="${mode_dir}/infer.log"
      eval_log="${mode_dir}/eval.log"

      cmd=(
        "${PYTHON_BIN}" logo/infer_logo.py
        --model_type "${MODEL_TYPE}"
        --gpu_id "${GPU_ID}"
        --device_map "${DEVICE_MAP}"
        --test_data "${test_file}"
        --signal_type "${signal_type}"
        --merge_method "${MERGE_METHOD}"
        --inference_mode "${mode}"
        --lora_type app
        --lora_pool "${APP_POOL}"
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

      echo "[INFO] Running ${mode} inference for ${app} (signal_type=${signal_type}, merge_method=${MERGE_METHOD})..."
      if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" 2>&1 | tee "${infer_log}"; then
        echo "[ERROR] ${mode} inference failed for ${app}, signal_type=${signal_type}. See ${infer_log}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
      fi

      result_jsonl="$(extract_jsonl_from_infer_log "${infer_log}")"
      if [[ -n "${result_jsonl}" && ! -f "${result_jsonl}" ]]; then
        echo "[WARNING] Parsed JSONL does not exist, will fallback to latest in output dir: ${result_jsonl}"
        result_jsonl=""
      fi
      if [[ -z "${result_jsonl}" ]]; then
        echo "[WARNING] Could not parse infer JSONL from current run log; fallback to latest file in ${mode_dir}/episode_outputs"
        result_jsonl="$(find_latest_jsonl "${mode_dir}/episode_outputs")"
      fi
      if [[ -z "${result_jsonl}" ]]; then
        echo "[ERROR] ${mode} result jsonl not found for ${app}, signal_type=${signal_type}."
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
      fi

      echo "[INFO] Evaluating ${mode} result: ${result_jsonl}"
      if ! "${PYTHON_BIN}" evaluation/evaluate_inference.py --results_path "${result_jsonl}" 2>&1 | tee "${eval_log}"; then
        echo "[ERROR] evaluation failed for ${app} ${mode}, signal_type=${signal_type}. See ${eval_log}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
      fi

      if [[ "${mode}" == "step" ]]; then
        step_acc="$(extract_metric_from_eval "${eval_log}" "Step-level Accuracy")"
        step_jsonl="${result_jsonl}"
      else
        episode_acc="$(extract_metric_from_eval "${eval_log}" "Episode-level Accuracy")"
        episode_jsonl="${result_jsonl}"
      fi
    done

    echo -e "${app}\t${signal_type}\t${step_acc}\t${episode_acc}\t${step_jsonl}\t${episode_jsonl}" >> "${SUMMARY_TSV}"
    echo "${app},${signal_type},${step_acc},${episode_acc},${step_jsonl},${episode_jsonl}" >> "${SUMMARY_CSV}"
  done
done

echo ""
echo "==========================================="
echo "Summary Table"
echo "==========================================="
printf "%-18s | %-20s | %-14s | %-16s\n" "app" "signal_type" "step_accuracy" "episode_accuracy"
printf "%-18s-+-%-20s-+-%-14s-+-%-16s\n" "------------------" "--------------------" "--------------" "----------------"
tail -n +2 "${SUMMARY_TSV}" | while IFS=$'\t' read -r app signal_type step_acc episode_acc step_jsonl episode_jsonl; do
  printf "%-18s | %-20s | %-14s | %-16s\n" "${app}" "${signal_type}" "${step_acc}" "${episode_acc}"
done

echo "==========================================="
echo "Detailed TSV: ${SUMMARY_TSV}"
echo "Detailed CSV: ${SUMMARY_CSV}"
echo "==========================================="
if (( FAIL_COUNT > 0 )); then
  echo "[ERROR] Batch evaluation finished with ${FAIL_COUNT} failed sub-runs."
  exit 1
fi
