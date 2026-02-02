# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LoGO-internVL2 implements the **LoRA on the Go (LOGO)** paper - a dynamic, instance-level LoRA selection and merging framework for vision-language models. Based on InternVL2-2B, it automatically selects the most relevant LoRAs for each input without manual specification.

**Core Pipeline:**
1. **Signal Extraction**: Compute activation signals from LoRA projections in target transformer block
2. **Top-K Selection**: Select K most relevant LoRAs based on signal strength
3. **Mixture Fusion**: Merge selected LoRAs with weights proportional to their signals

## Key Commands

```bash
# Environment setup
conda activate LoGO
export CUDA_VISIBLE_DEVICES=5
export MAX_PIXELS=150000
export MAX_NUM=9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Full evaluation (App + Category)
python logo/evaluate_logo.py --gpu 5 --top_k 3 --signal_type norm

# App-level only
python logo/evaluate_logo.py --gpu 5 --top_k 3 --signal_type norm --app_only

# Category-level only
python logo/evaluate_logo.py --gpu 5 --top_k 3 --signal_type norm --category_only

# Single dataset inference
python logo/infer_logo.py --test_data data/Val_100.jsonl --top_k 3 --signal_type norm

# Shell script (GPU_ID, TEST_DATA, TOP_K, SIGNAL_TYPE, MERGE_METHOD)
bash logo/run_logo.sh 5 data/Val_100.jsonl 3 norm mixture

# Single LoRA baseline
bash logo/run_single_lora.sh 5 app_lora_youtube data/Val_100.jsonl
bash logo/run_single_lora.sh --list  # list available LoRAs

# Evaluate results
python evaluation/test_swift.py --data_path output/logo_results_*.jsonl
```

## Architecture

```
logo/
├── logo_engine.py       # Main engine: LOGOEngine, load_lora_configs(), create_logo_engine()
├── signal_extractor.py  # Signal extraction: compute_norm_signal(), compute_entropy_signal()
├── infer_logo.py        # Inference entry point
└── evaluate_logo.py     # Evaluation harness

config/
├── app_loras_config_internvl2.json      # 14 app-level LoRAs
└── category_loras_config_internvl2.json # 5 category-level LoRAs

swift/tuners/
├── lora_layers.py       # MODIFIED: mixture/fusion logic for weighted LoRA merging
└── peft.py              # MODIFIED: supports merging_type and lora_mapping parameters

evaluation/
└── test_swift.py        # Accuracy calculation using TF-IDF similarity
```

## Key Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--top_k` | 5 | 1-20+ | Number of top LoRAs to select |
| `--signal_type` | norm | norm, entropy, embedding, uniform | Signal extraction method |
| `--merge_method` | mixture | mixture, add_weighted_adapter | LoRA merging approach |
| `--target_block_idx` | -1 | -1 to N-1 | Transformer block for signals (-1 = last) |
| `--no_baseline_calibration` | False | - | Disable baseline signal calibration |

## Data Formats

**LoRA Config (JSON):**
```json
[{"lora_name": "app_lora_youtube", "lora_path": "/path/to/checkpoint", "embedding_path": "data/embeddings/youtube_emb.npy"}]
```

**Test Data (JSONL):**
```json
{"images": ["/path/to/image.png"], "query": "<image>\nInstruction", "response": "Expected action", "episode_id": "001234"}
```

## Known Issues

1. **LoRA Selection Bias**: When using ratio calibration (`signal / baseline`), LoRAs with small baseline values are over-selected. `app_lora_decathlon` (baseline=0.407) is always selected as Top-1 regardless of input.

2. **Memory**: Loading all 19 LoRAs requires 48GB+ VRAM. Use environment variables above to prevent OOM.

## Model Paths

- **InternVL2-2B**: `/home/hmpiao/hmpiao/InternVL2-2B-ModelScope/OpenGVLab/InternVL2-2B`
- **Qwen2-VL-7B**: `/home/hmpiao/hmpiao/Qwen2-VL-7B-Instruct`
