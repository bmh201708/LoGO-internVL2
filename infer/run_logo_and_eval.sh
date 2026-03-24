#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults (can be overridden by env vars).
# bash infer/run_logo_and_eval.sh --merge_method mixture --inference_mode step  2>&1 | tee  /data0/piaohongming/jinyike/LoGO-internVL2/test.log
# nohup bash infer/run_logo_and_eval.sh --gpu_id 4 --model_type qwen2-vl-7b-instruct --test_data /data0/piaohongming/jinyike/data/data-test/app/ebay_train.jsonl --signal_type entropy --merge_method mixture --inference_mode step --lora_type app --lora_pool amazon,clock,ebay,etsy,flipkart,google_drive,reminder,youtube --top_k 3 --target_block_idx -1 --token_position last --temperature 0.0 --seed 42 --no_baseline_calibration --num_samples 20 --output_dir /data0/piaohongming/jinyike/LoGO-internVL2/output/batch_eval_logo/20260322-140309/ebay/entropy/step > rerun_step.nohup.log 2>&1 &
PYTHON_BIN="${PYTHON_BIN:-/data0/piaohongming/envs/LoGO/bin/python}"
GPU_ID="${GPU_ID:-0}"
MODEL_TYPE="${MODEL_TYPE:-qwen2-vl-2b-instruct}"
TEST_DATA="${TEST_DATA:-/data0/piaohongming/jinyike/data/data-test/app/amazon_train.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SIGNAL_TYPE="${SIGNAL_TYPE:-norm}"
MERGE_METHOD="${MERGE_METHOD:-add_weighted_adapter}"
LORA_TYPE="${LORA_TYPE:-app}"
TOP_K="${TOP_K:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/compat_check}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
INFERENCE_MODE="${INFERENCE_MODE:-episode}"
LORA_POOL="${LORA_POOL:-}"
TARGET_BLOCK_IDX="${TARGET_BLOCK_IDX:--1}"
TOKEN_POSITION="${TOKEN_POSITION:-last}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
USE_BASELINE_CALIBRATION="${USE_BASELINE_CALIBRATION:-0}"

# Candidate pools (can be edited directly).
APP=(
  amazon clock ebay etsy flipkart google_drive reminder youtube
)
CATEGORY=(entertainment lives office shopping traveling)

# Allow overriding key script vars via CLI flags as well.
# This keeps printed mode/output_dir and post-eval metric selection aligned
# with the actual infer_logo.py invocation.
CLI_ARGS=("$@")
CLI_LORA_POOL_SET=0
FORWARD_ARGS=()
i=0
while (( i < ${#CLI_ARGS[@]} )); do
  arg="${CLI_ARGS[$i]}"
  arg_lc="${arg,,}"
  case "${arg_lc}" in
    --inference_mode|--output_dir|--lora_type|--lora_pool|--signal_type|--top_k|--merge_method|--target_block_idx|--token_position|--temperature|--seed|--num_samples|--gpu_id|--device_map|--model_type|--test_data)
      if (( i + 1 >= ${#CLI_ARGS[@]} )); then
        echo "[ERROR] Missing value for argument: ${arg}"
        exit 2
      fi
      val="${CLI_ARGS[$((i + 1))]}"
      case "${arg_lc}" in
        --inference_mode) INFERENCE_MODE="${val}" ;;
        --output_dir) OUTPUT_DIR="${val}" ;;
        --lora_type) LORA_TYPE="${val}" ;;
        --lora_pool) LORA_POOL="${val}"; CLI_LORA_POOL_SET=1 ;;
        --signal_type) SIGNAL_TYPE="${val}" ;;
        --top_k) TOP_K="${val}" ;;
        --merge_method) MERGE_METHOD="${val}" ;;
        --target_block_idx) TARGET_BLOCK_IDX="${val}" ;;
        --token_position) TOKEN_POSITION="${val}" ;;
        --temperature) TEMPERATURE="${val}" ;;
        --seed) SEED="${val}" ;;
        --num_samples) NUM_SAMPLES="${val}" ;;
        --gpu_id) GPU_ID="${val}" ;;
        --device_map) DEVICE_MAP="${val}" ;;
        --model_type) MODEL_TYPE="${val}" ;;
        --test_data) TEST_DATA="${val}" ;;
      esac
      i=$((i + 2))
      ;;
    --inference_mode=*|--output_dir=*|--lora_type=*|--lora_pool=*|--signal_type=*|--top_k=*|--merge_method=*|--target_block_idx=*|--token_position=*|--temperature=*|--seed=*|--num_samples=*|--gpu_id=*|--device_map=*|--model_type=*|--test_data=*)
      val="${arg#*=}"
      case "${arg_lc}" in
        --inference_mode=*) INFERENCE_MODE="${val}" ;;
        --output_dir=*) OUTPUT_DIR="${val}" ;;
        --lora_type=*) LORA_TYPE="${val}" ;;
        --lora_pool=*) LORA_POOL="${val}"; CLI_LORA_POOL_SET=1 ;;
        --signal_type=*) SIGNAL_TYPE="${val}" ;;
        --top_k=*) TOP_K="${val}" ;;
        --merge_method=*) MERGE_METHOD="${val}" ;;
        --target_block_idx=*) TARGET_BLOCK_IDX="${val}" ;;
        --token_position=*) TOKEN_POSITION="${val}" ;;
        --temperature=*) TEMPERATURE="${val}" ;;
        --seed=*) SEED="${val}" ;;
        --num_samples=*) NUM_SAMPLES="${val}" ;;
        --gpu_id=*) GPU_ID="${val}" ;;
        --device_map=*) DEVICE_MAP="${val}" ;;
        --model_type=*) MODEL_TYPE="${val}" ;;
        --test_data=*) TEST_DATA="${val}" ;;
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
      FORWARD_ARGS+=("${CLI_ARGS[$i]}")
      i=$((i + 1))
      ;;
  esac
done

if [[ "${CLI_LORA_POOL_SET}" -eq 0 && -z "${LORA_POOL}" ]]; then
  case "${LORA_TYPE}" in
    app)
      LORA_POOL="$(IFS=,; echo "${APP[*]}")"
      ;;
    category)
      LORA_POOL="$(IFS=,; echo "${CATEGORY[*]}")"
      ;;
    all)
      LORA_POOL="$(IFS=,; echo "${APP[*]},${CATEGORY[*]}")"
      ;;
    *)
      echo "[WARNING] Unknown LORA_TYPE=${LORA_TYPE}; defaulting empty lora pool."
      LORA_POOL=""
      ;;
  esac
fi

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

echo "==========================================="
echo "LOGO Inference + Evaluation"
echo "==========================================="
echo "PYTHON_BIN : ${PYTHON_BIN}"
echo "GPU_ID     : ${GPU_ID}"
echo "MODEL_TYPE : ${MODEL_TYPE}"
echo "TEST_DATA  : ${TEST_DATA}"
echo "MODE       : ${INFERENCE_MODE}"
echo "LORA_TYPE  : ${LORA_TYPE}"
echo "LORA_POOL  : ${LORA_POOL}"
echo "TOP_K      : ${TOP_K}"
echo "SIGNAL     : ${SIGNAL_TYPE}"
echo "MERGE      : ${MERGE_METHOD}"
echo "TGT_BLOCK  : ${TARGET_BLOCK_IDX}"
echo "TOKEN_POS  : ${TOKEN_POSITION}"
echo "TEMP       : ${TEMPERATURE}"
echo "SEED       : ${SEED}"
echo "BASELINE   : ${USE_BASELINE_CALIBRATION}"
echo "OUTPUT_DIR : ${OUTPUT_DIR}"
echo "==========================================="

cd "${REPO_ROOT}"

INFER_CMD=(
  "${PYTHON_BIN}" logo/infer_logo.py
  --model_type "${MODEL_TYPE}"
  --gpu_id "${GPU_ID}"
  --device_map "${DEVICE_MAP}"
  --test_data "${TEST_DATA}"
  --num_samples "${NUM_SAMPLES}"
  --signal_type "${SIGNAL_TYPE}"
  --merge_method "${MERGE_METHOD}"
  --inference_mode "${INFERENCE_MODE}"
  --lora_type "${LORA_TYPE}"
  --lora_pool "${LORA_POOL}"
  --top_k "${TOP_K}"
  --target_block_idx "${TARGET_BLOCK_IDX}"
  --token_position "${TOKEN_POSITION}"
  --temperature "${TEMPERATURE}"
  --seed "${SEED}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${USE_BASELINE_CALIBRATION}" != "1" ]]; then
  INFER_CMD+=(--no_baseline_calibration)
fi

INFER_CMD+=("${FORWARD_ARGS[@]}")
INFER_LOG="$(mktemp)"
if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" "${INFER_CMD[@]}" 2>&1 | tee "${INFER_LOG}"; then
  echo "[ERROR] Inference failed. See log: ${INFER_LOG}"
  exit 1
fi

RESULT_DIR="${OUTPUT_DIR}/episode_outputs"
if [[ ! -d "${RESULT_DIR}" ]]; then
  echo "[ERROR] Result directory not found: ${RESULT_DIR}"
  exit 1
fi

RESULT_JSONL="$(extract_jsonl_from_infer_log "${INFER_LOG}")"
if [[ -n "${RESULT_JSONL}" && ! -f "${RESULT_JSONL}" ]]; then
  echo "[WARNING] JSONL from infer log does not exist: ${RESULT_JSONL}"
  RESULT_JSONL=""
fi
if [[ -z "${RESULT_JSONL}" ]]; then
  echo "[WARNING] Could not parse infer JSONL from current run log; fallback to latest file in ${RESULT_DIR}"
  RESULT_JSONL="$(
  find "${RESULT_DIR}" -maxdepth 1 -type f -name 'retriever_results_*.jsonl' -printf '%T@ %p\n' \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-
)"
fi

if [[ -z "${RESULT_JSONL}" ]]; then
  echo "[ERROR] No result JSONL found under: ${RESULT_DIR}"
  exit 1
fi

echo "Latest JSONL: ${RESULT_JSONL}"
echo "==========================================="
echo "Running Evaluation"
echo "==========================================="

EVAL_LOG="$(mktemp)"
"${PYTHON_BIN}" evaluation/evaluate_inference.py --results_path "${RESULT_JSONL}" | tee "${EVAL_LOG}"

echo "==========================================="
echo "Selected Metric (${INFERENCE_MODE})"
echo "==========================================="
if [[ "${INFERENCE_MODE}" == "step" ]]; then
  METRIC_LINE="$(grep -E 'Step-level Accuracy' "${EVAL_LOG}" | head -n 1 || true)"
  if [[ -n "${METRIC_LINE}" ]]; then
    echo "${METRIC_LINE}"
  else
    echo "[WARNING] Step accuracy line not found in evaluation output."
  fi
elif [[ "${INFERENCE_MODE}" == "episode" ]]; then
  METRIC_LINE="$(grep -E 'Episode-level Accuracy' "${EVAL_LOG}" | head -n 1 || true)"
  if [[ -n "${METRIC_LINE}" ]]; then
    echo "${METRIC_LINE}"
  else
    echo "[WARNING] Episode accuracy line not found in evaluation output."
  fi
else
  echo "[WARNING] Unknown INFERENCE_MODE=${INFERENCE_MODE}; skip metric filtering."
fi

rm -f "${EVAL_LOG}"
rm -f "${INFER_LOG}"
