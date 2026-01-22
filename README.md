# LoGO-internVL2

基于 InternVL2-2B 复现 **LoRA on the Go (LOGO)** 论文的动态 LoRA 选择与融合框架。

## 📖 项目简介

LOGO (LoRA on the Go) 是一种实例级别的动态 LoRA 选择和融合方法，能够根据输入自动选择最相关的 LoRAs 并加权融合，无需人工指定使用哪个 LoRA。

### 核心思想

1. **信号提取 (Signal Extraction)**: 对每个输入，计算各个 LoRA 在目标 Transformer Block 的激活信号
2. **Top-K 选择**: 根据信号强度选择 Top-K 个最相关的 LoRAs
3. **加权融合 (Mixture)**: 将选中的 LoRAs 按权重加权融合输出

### 项目结构

```
LoGO-internVL2/
├── logo/                          # LOGO 核心实现
│   ├── logo_engine.py             # LOGO 引擎（选择、融合逻辑）
│   ├── signal_extractor.py        # 信号提取器
│   ├── infer_logo.py              # LOGO 推理脚本
│   ├── evaluate_logo.py           # LOGO 评估脚本
│   ├── run_logo.sh                # LOGO 运行脚本
│   └── run_single_lora.sh         # 单 LoRA 基线脚本
├── config/                        # LoRA 配置文件
│   ├── app_loras_config_internvl2.json      # App 级别 LoRA 配置
│   └── category_loras_config_internvl2.json # Category 级别 LoRA 配置
├── data/                          # 测试数据
│   ├── test_data_by_app/          # App 级别测试集
│   └── test_data_by_category/     # Category 级别测试集
├── evaluation/                    # 评估工具
│   └── test_swift.py              # 准确率计算脚本
├── swift/                         # MS-Swift 框架（已修改支持 Mixture 模式）
└── output/                        # 输出结果
```

## 🔬 论文复现

### 复现的核心功能

| 论文章节 | 实现状态 | 文件位置 |
|---------|---------|---------|
| Signal Extraction (Eq. 2-3) | ✅ L2 Norm / Entropy | `signal_extractor.py` |
| Top-K Selection (Eq. 4) | ✅ | `logo_engine.py` |
| Weight Normalization (Eq. 5) | ✅ Softmax | `logo_engine.py` |
| Mixture Fusion (Eq. 6) | ✅ Output-level | `swift/tuners/lora_layers.py` |
| Baseline Calibration | ✅ Ratio / Difference | `signal_extractor.py` |

### 信号类型

- **norm**: L2 范数（默认），计算 LoRA 投影输出的激活强度
- **entropy**: 熵的倒数，用于衡量激活分布的确定性
- **uniform**: 均匀权重，作为基线对比

## 🚀 快速开始

### 环境配置

```bash
conda activate LoGO
cd LoGO-internVL2
```

### 运行 LOGO 评估

```bash
# 完整评估（App + Category）
python logo/evaluate_logo.py --gpu 5 --top_k 3 --signal_type norm

# 仅评估 App 级别
python logo/evaluate_logo.py --gpu 5 --top_k 3 --signal_type norm --app_only

# 仅评估 Category 级别
python logo/evaluate_logo.py --gpu 5 --top_k 3 --signal_type norm --category_only
```

### 使用 Shell 脚本

```bash
# LOGO 推理（单个数据集）
bash logo/run_logo.sh 5 data/Val_100.jsonl 3 norm mixture

# 参数说明:
# $1: GPU_ID
# $2: TEST_DATA
# $3: TOP_K
# $4: SIGNAL_TYPE (norm / entropy / uniform)
# $5: MERGE_METHOD (mixture)
```

### 单 LoRA 基线对比

```bash
# 列出可用的 LoRA
bash logo/run_single_lora.sh --list

# 使用指定的 LoRA 进行推理
bash logo/run_single_lora.sh 5 app_lora_youtube data/Val_100.jsonl
```

## 📊 评估结果

### App-level Results (14 App LoRAs)

| App | Step Accuracy | Episode Accuracy |
|-----|---------------|------------------|
| clock | 100.0% | 100.0% |
| gmail | 100.0% | 100.0% |
| reminder | 93.33% | 92.86% |
| google_drive | 90.0% | 90.0% |
| amazon | 86.67% | 86.67% |
| **Average** | **76.07%** | **76.04%** |

### Category-level Results (5 Category LoRAs)

| Category | Step Accuracy | Episode Accuracy |
|----------|---------------|------------------|
| shopping | 78.12% | 77.42% |
| lives | 76.19% | 75.0% |
| traveling | 73.33% | 66.67% |
| **Average** | **72.55%** | **73.66%** |

## ⚠️ 已知问题与局限性

### 1. LoRA 选择偏差问题

**问题描述**: 当使用 Ratio 比值校准时，baseline 激活值小的 LoRA 会被过度选择。

**具体表现**:
- `app_lora_decathlon` 的 baseline 最小 (0.407)
- 无论输入什么数据，经过比值校准后它总是被选为 Top-1
- 导致选择机制失效，无法正确选择与输入相关的 LoRA

**Baseline 值分布**:
```
app_lora_amazon:     1.735 (最大)
app_lora_gmail:      1.414
app_lora_ebay:       1.212
...
app_lora_decathlon:  0.407 (最小) ← 总是被过度选中
```

**根本原因**: `calibrated_signal = signal / baseline`，当 baseline 很小时，校准后的值会被放大。

### 2. 可能的解决方案（待实现）

| 方案 | 公式 | 状态 |
|------|------|------|
| 差值校准 | `signal - baseline` | 🔄 待测试 |
| 相对增益率 | `(signal - baseline) / baseline` | 🔄 待测试 |
| 排名法 | `rank(signal - baseline)` | 🔄 待测试 |
| Z-score | `(signal - μ) / σ` | 🔄 待测试 |
| 截断比值 | `min(signal/baseline, max_ratio)` | 🔄 待测试 |

### 3. 显存使用

LOGO 需要同时加载所有 LoRAs，显存占用较高：
- 14 个 App LoRAs + 5 个 Category LoRAs
- 推荐使用 48GB+ 显存的 GPU

**防止 OOM 的环境变量**:
```bash
export MAX_PIXELS=150000
export MAX_NUM=9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 📁 配置文件格式

### LoRA 配置 (JSON)

```json
[
    {
        "lora_name": "app_lora_youtube",
        "lora_path": "/path/to/lora/checkpoint",
        "embedding_path": "data/embeddings/youtube_emb.npy",
        "description": "InternVL2-2B LoRA for YouTube app"
    }
]
```

### 测试数据格式 (JSONL)

```json
{
    "images": ["/path/to/image1.png", "/path/to/image2.png"],
    "query": "<image>\n<image>\nUser instruction here",
    "response": "Expected action sequence",
    "episode_id": "001234"
}
```

## 🔧 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--top_k` | 3 | 选择 Top-K 个 LoRAs |
| `--signal_type` | norm | 信号类型 (norm/entropy/uniform) |
| `--merge_method` | mixture | 融合方法 (mixture) |
| `--target_block_idx` | -1 | 目标 Transformer Block (-1 = 最后一层) |
| `--no_baseline_calibration` | False | 禁用基线校准 |

## 📚 参考

- **论文**: [LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging](LoRA%20on%20the%20Go.pdf)
- **基础模型**: InternVL2-2B
- **框架**: MS-Swift (修改版，支持 Mixture 模式)

## 📝 License

本项目仅用于学术研究。
