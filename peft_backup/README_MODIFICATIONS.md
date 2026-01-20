# PEFT 库修改说明

本 PEFT 库（v0.18.0）已经过修改，支持 **LoraRetriever** 和 **LOGO** 项目的高级功能。

## 新增功能

### LoraRetriever 功能
✅ **Fusion 合并模式** - 先合并 LoRA 权重再计算（快速）  
✅ **Mixture 混合模式** - 先计算再混合（准确）  
✅ **Batch 级动态推理** - 每个样本使用不同的 LoRA 组合  

### LOGO 功能 (新增)
✅ **单次前向传播信号提取** - 一次前向传播获取所有 LoRA 的单独输出  
✅ **目标层过滤** - 仅存储指定层的输出，避免内存溢出  
✅ **向后兼容** - 新参数可选，不影响现有代码  

---

## 快速测试

```bash
# 测试 LoraRetriever 功能
cd /home/hmpiao/hmpiao/jinyike/lorahub-for-fedmabench/peft
python test_lora_merging.py

# 测试 LOGO 单次前向传播
cd /home/hmpiao/hmpiao/jinyike/lorahub-for-fedmabench
python LoGO/tests/test_core.py
```

---

## LOGO 使用示例

```python
# 使用单次前向传播获取每个 LoRA 的单独输出
outputs = model(
    **inputs,
    merging_type="mixture",
    lora_mapping=uniform_mapping,
    return_individual_outputs=True,   # 启用单独输出返回
    logo_target_layer=27,             # 仅在第 27 层存储输出
)

# 输出存储在目标层模块的属性中
for name, module in model.named_modules():
    if hasattr(module, '_logo_individual_outputs'):
        individual_outputs = module._logo_individual_outputs
        # individual_outputs 形状: (batch, seq_len, num_loras, hidden_dim)
        break
```

---

## 修改的文件

### 1. `tuners/lora/layer.py`

| 行号 | 修改内容 |
|------|----------|
| 41 | 添加 `return_individual_outputs`, `logo_target_layer` 到 `VARIANT_KWARG_KEYS` |
| 853-863 | 添加 `individual_outputs` 存储逻辑（仅匹配目标层时存储） |

### 2. `tuners/lora/model.py`

| 行号 | 修改内容 |
|------|----------|
| 69-80 | 更新 `_merging_params_pre_forward_hook` 支持新参数 |
| 419-431 | 更新 hook 注册传递 `logo_target_layer` |
| 365-376 | 在 `_enable_peft_forward_hooks` 中提取 `return_individual_outputs` |

---

## 参数说明

### 原有参数 (LoraRetriever)

| 参数 | 类型 | 说明 |
|------|------|------|
| `merging_type` | `str` | `'fusion'` / `'mixture'` / `None` |
| `lora_mapping` | `Tensor` | `(batch, num_loras)` 权重矩阵 |

### 新增参数 (LOGO)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `return_individual_outputs` | `bool` | `None` | 启用时将单独输出存储到模块属性 |
| `logo_target_layer` | `int` | `None` | 仅在指定层索引存储输出 |

---

## 向后兼容性

| 场景 | 行为 |
|------|------|
| 不传任何新参数 | 完全兼容原始 PEFT |
| 仅传 `merging_type` + `lora_mapping` | LoraRetriever 模式 |
| 传 `return_individual_outputs=True` | LOGO 模式（存储单独输出） |

---

## 测试状态

✅ 语法检查通过  
✅ LoraRetriever 功能测试通过  
✅ LOGO 单次前向传播测试通过  
✅ 向后兼容性验证通过  
