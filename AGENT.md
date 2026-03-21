# Repository Agent Guide (LoGO-internVL2)

## 1) Purpose
This repo implements **LoRA on the Go (LoGO)** for VLM inference, with dynamic LoRA selection and weighted composition.

## 2) Environment
- Expected conda env name in docs: `LoGO`
- Typical startup:
  - `conda activate LoGO`
  - `cd /data1/hmpiao/jinyike/LoGO-internVL2`
- Important: `logo/evaluate_logo_qwen2vl.sh` does **not** activate conda automatically. It uses the currently active Python environment.

## 3) Key Entry Points
- `logo/evaluate_logo_qwen2vl.sh`: shell entry for Qwen2-VL evaluation.
- `logo/evaluate_logo.py`: batch evaluation driver over app/category datasets.
- `logo/infer_logo.py`: single-run inference pipeline.
- `logo/logo_engine.py`: LoRA loading, signal extraction, top-k selection, and merge orchestration.
- `logo/signal_extractor.py`: signal computation (`norm`, `entropy`, `uniform`) and weight normalization.

## 4) Merge Modes
- `mixture` (paper-recommended): output-level weighted sum at forward time.
  - Runtime formula: `base(x) + sum_i w_i * LoRA_i(x)`
  - Context wiring in `swift/llm/utils/utils.py` with `merging_type='mixture'`.
  - Core forward patch in `swift/tuners/lora_layers.py` and `swift/tuners/peft.py`.
- `add_weighted_adapter`: parameter-level merge through PEFT/Swift API.
  - Implemented via `model.add_weighted_adapter(...)` in `logo/logo_engine.py`.

## 5) Candidate Pool and Dataset Scope
- Use `--lora_type all|app|category` in `logo/infer_logo.py`.
- Dataset convention:
  - `data/data-test` is the **test set** root.
- For strict isolation:
  - app-only: `--app_config <app_json>` + `--category_config /dev/null`
  - category-only: `--app_config /dev/null` + `--category_config <category_json>`

## 6) peft vs peft_backup
- Main runtime imports `peft` (standard module name), not `peft_backup`.
- `peft_backup/` is a backup/experimental tree and is not referenced by default execution path.
- If behavior looks unusual, verify:
  - `python -c "import peft; print(peft.__file__)"`

## 7) Practical Run Examples
- Full eval (Qwen2-VL-7B):  
  `bash logo/evaluate_logo_qwen2vl.sh 5 3 all`
- App-only quick check:  
  `python logo/infer_logo.py --model_type qwen2-vl-7b-instruct --lora_type app --num_samples 3`

## 8) Migration Plan (Align with LoraRetriever Workflow)
Use this checklist to refactor LoGO with the same strategy used in LoraRetriever.

### A. LoRA config refactor (model-specific + strict scope)
- Update model-to-config mapping in `logo/infer_logo.py`:
  - Split app/category config by model (`internvl2`, `qwen2b`, `qwen7b` if needed).
  - Support absolute path override for `--app_config` / `--category_config`.
- Add strict gate by `--lora_type`:
  - `app`: force `category_config=/dev/null`
  - `category`: force `app_config=/dev/null`
- Add optional candidate pool control:
  - App whitelist tuple/list (like `APP=(...)`) to constrain retrievable candidates.

### B. Evaluation dataset refactor
- In `logo/evaluate_logo.py` and shell wrappers:
  - Separate app test root and category test root.
  - If `--lora_type app`, dataset must be app; if `category`, dataset must be category.
  - On mismatch, exit immediately (`exit 1`) with explicit error.

### C. Multi-turn data format support
- Extend `logo/infer_logo.py` input parser:
  - Support episode-style multi-turn JSONL (messages/history + images per turn).
  - Add two modes:
    - `step`: history uses ground truth assistant turns.
    - `episode`: history uses model previous outputs.
- Retrieval granularity:
  - Re-select LoRA **per turn** (not once per episode).
- History media policy:
  - Include history images in prompt context.
  - If image count exceeds limit, drop oldest images first.

### D. Heterogeneous-rank LoRA compatibility (mixture + fusion)
- `mixture`:
  - Keep output-level weighted sum (`base + sum_i w_i*LoRA_i(x)`), naturally rank-agnostic.
- `fusion` (parameter-level):
  - Current `add_weighted_adapter` may fail with mixed rank under linear mode.
  - Add mixed-rank path (weighted-delta + SVD, or equivalent) to avoid same-rank constraint.
  - Keep fallback and clear logs when merge degrades to single adapter.

### E. Evaluation method alignment
- Produce outputs compatible with evaluation scripts used in FedMABench format.
- Report both:
  - Step accuracy (single-turn correctness)
  - Episode accuracy (all turns in episode must pass)
- Add one-click scripts (ID/OOD, app/category) with strict type checks and clear run IDs.

### F. Suggested implementation order
1. Config + lora_type strict gating.
2. Multi-turn parser + step/episode inference logic.
3. Per-turn selection and history image truncation policy.
4. Mixed-rank fusion compatibility.
5. Evaluation output schema + batch evaluation scripts.

### G. Acceptance criteria
- `--lora_type app` run logs contain no category LoRA load/selection.
- `--lora_type category` run logs contain no app LoRA load/selection.
- Multi-turn step/episode both run on the same dataset and produce distinct metrics.
- Mixed-rank LoRA sets run in both `mixture` and `fusion` without rank mismatch crash.
