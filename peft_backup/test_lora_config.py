#!/usr/bin/env python3
"""测试加载 LoRA 并手动修改配置"""

import torch
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peft import PeftModel, LoraConfig
from transformers import Qwen2VLForConditionalGeneration

# 加载模型
print("加载模型...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print(f"✅ 模型加载成功: {type(model).__name__}")

# 读取 LoRA 配置
lora_path = "/home/hmpiao/hmpiao/xuerong/FedMABench/lora_category/category_lora_entertainment"
config_path = os.path.join(lora_path, "adapter_config.json")

print(f"\n读取 LoRA 配置: {config_path}")
with open(config_path) as f:
    config_dict = json.load(f)

print(f"原始 target_modules: {config_dict['target_modules']}")

# 修改 target_modules 为更具体的匹配
# 从 LoRA 权重键名我们知道实际的模块是在 model.language_model.layers.X...
# 所以我们改为只匹配 language_model 下的模块
config_dict['target_modules'] = r".*language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"

print(f"修改后  target_modules: {config_dict['target_modules']}")

# 从字典创建 LoraConfig
lora_config = LoraConfig(**config_dict)

# 尝试加载 PEFT 模型
print(f"\n加载 PEFT 模型...")
try:
    peft_model = PeftModel.from_pretrained(
        model,
        lora_path,
        adapter_name="test",
        key_mapping={"model.model": "model"}
    )
    print("✅ LoRA 加载成功！")
    print(f"Active adapters: {peft_model.active_adapters}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()
