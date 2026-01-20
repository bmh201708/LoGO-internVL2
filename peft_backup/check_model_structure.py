#!/usr/bin/env python3
"""检查 Qwen2-VL 模型结构和 LoRA 配置"""

import torch
from transformers import AutoModel
import json

# 加载模型
print("加载 Qwen2-VL 模型...")
model = AutoModel.from_pretrained(
    "/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("\n模型结构（前50个模块）:")
for i, (name, module) in enumerate(model.named_modules()):
    if i < 50:
        print(f"{name}: {type(module).__name__}")

print("\n\n所有包含 'attn' 或 'mlp' 的模块:")
for name, module in model.named_modules():
    if 'attn' in name.lower() or 'mlp' in name.lower():
        print(f"{name}: {type(module).__name__}")

# 检查 LoRA 配置
print("\n\nLoRA 配置:")
with open("/home/hmpiao/hmpiao/xuerong/FedMABench/lora_category/category_lora_entertainment/adapter_config.json") as f:
    config = json.load(f)
    print(json.dumps(config, indent=2))
