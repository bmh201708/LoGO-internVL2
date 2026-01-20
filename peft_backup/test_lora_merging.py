#!/usr/bin/env python3
"""测试 LoRA Fusion/Mixture 合并功能 - 使用真实的 Qwen2-VL 模型和 Category LoRAs"""

import torch
import sys
import os
import json
import tempfile
import shutil

# 添加当前目录的父目录到 path（peft 的父目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 模型和 LoRA 路径配置
BASE_MODEL_PATH = "/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct"
LORA_BASE_PATH = "/home/hmpiao/hmpiao/xuerong/FedMABench/lora_category"

LORA_ADAPTERS = [
    "category_lora_entertainment",
    "category_lora_office", 
    "category_lora_traveling",
    "category_lora_lives",
    "category_lora_shopping"
]


def create_temp_lora_with_fixed_config(original_lora_path):
    """创建临时 LoRA 目录，修正 target_modules 配置"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="lora_temp_")
    
    # 复制权重文件
    for filename in ["adapter_model.safetensors", "adapter_model.bin"]:
        src = os.path.join(original_lora_path, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(temp_dir, filename))
    
    # 读取并修改配置
    config_path = os.path.join(original_lora_path, "adapter_config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    # 修改 target_modules 为更精确的正则表达式
    # 原来: ^(model)(?!.*(lm_head|output|emb|wte|shared)).*
    # 新的: 只匹配 language_model 下的层
    config['target_modules'] = r".*language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    
    # 保存修改后的配置到临时目录
    with open(os.path.join(temp_dir, "adapter_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    return temp_dir


def test_load_model_and_loras():
    """测试加载基础模型和多个 LoRA 适配器"""
    print("=" * 80)
    print("测试 1: 加载 Qwen2-VL 模型和 Category LoRAs")
    print("=" * 80)
    
    temp_dirs = []
    
    try:
        from peft import PeftModel
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        
        # 加载基础模型
        print(f"\n🔄 加载基础模型: {BASE_MODEL_PATH}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✅ 基础模型加载成功")
        print(f"   模型类型: {type(model).__name__}")
        print(f"   设备: {next(model.parameters()).device}")
        
        # 加载第一个 LoRA - 使用临时配置
        first_lora_path = os.path.join(LORA_BASE_PATH, LORA_ADAPTERS[0])
        print(f"\n🔄 加载第一个 LoRA: {LORA_ADAPTERS[0]}")
        print(f"   创建临时配置...")
        
        temp_lora_path = create_temp_lora_with_fixed_config(first_lora_path)
        temp_dirs.append(temp_lora_path)
        
        peft_model = PeftModel.from_pretrained(
            model,
            temp_lora_path,
            adapter_name=LORA_ADAPTERS[0],
            key_mapping={"model.model": "model"}
        )
        print(f"✅ LoRA 加载成功: {LORA_ADAPTERS[0]}")
        
        # 加载其他 LoRAs
        for lora_name in LORA_ADAPTERS[1:]:
            lora_path = os.path.join(LORA_BASE_PATH, lora_name)
            print(f"\n🔄 加载 LoRA: {lora_name}")
            
            temp_lora_path = create_temp_lora_with_fixed_config(lora_path)
            temp_dirs.append(temp_lora_path)
            
            peft_model.load_adapter(
                temp_lora_path,
                adapter_name=lora_name,
                key_mapping={"model.model": "model"}
            )
            print(f"✅ LoRA 加载成功: {lora_name}")
        
        # 检查 special_peft_forward_args
        print(f"\n📋 Special forward args: {peft_model.special_peft_forward_args}")
        
        # 所有 adapter 已经自动激活
        print(f"\n📋 Active adapters: {peft_model.active_adapters}")
        
        # 加载 processor
        print(f"\n🔄 加载 Processor")
        processor = Qwen2VLProcessor.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True
        )
        print(f"✅ Processor 加载成功")
        
        return peft_model, processor, temp_dirs, True
        
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理临时目录
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        return None, None, [], False


def test_fusion_mode(peft_model, processor):
    """测试 Fusion 模式"""
    print("\n" + "=" * 80)
    print("测试 2: Fusion 模式 - 多 LoRA 合并")
    print("=" * 80)
    
    try:
        # 准备测试输入
        test_messages = [
            {"role": "user", "content": "你好，请介绍一下自己。"}
        ]
        
        print(f"\n📝 测试输入: {test_messages[0]['content']}")
        
        text = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            return_tensors="pt",
        ).to(peft_model.device)
        
        print(f"✅ Processed 输入形状: {inputs['input_ids'].shape}")
        
        batch_size = inputs['input_ids'].shape[0]
        num_loras = len(LORA_ADAPTERS)
        
        # 创建 lora_mapping - 每个 LoRA 平均权重
        lora_mapping = torch.ones((batch_size, num_loras), dtype=torch.float32) / num_loras
        lora_mapping = lora_mapping.to(peft_model.device)
        
        print(f"\n📊 LoRA Mapping:")
        for i, lora_name in enumerate(LORA_ADAPTERS):
            print(f"   - {lora_name}: {lora_mapping[0, i]:.3f}")
        
        # Fusion 模式推理
        print(f"\n🚀 运行 Fusion 模式推理...")
        peft_model.eval()
        with torch.no_grad():
            outputs = peft_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                merging_type='fusion',
                lora_mapping=lora_mapping
            )
        
        print(f"✅ Fusion 模式成功")
        print(f"   输出形状: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Fusion 模式失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixture_mode(peft_model, processor):
    """测试 Mixture 模式"""
    print("\n" + "=" * 80)
    print("测试 3: Mixture 模式 - 多 LoRA 混合")
    print("=" * 80)
    
    try:
        test_messages = [
            {"role": "user", "content": "请介绍一个旅游景点。"}
        ]
        
        print(f"\n📝 测试输入: {test_messages[0]['content']}")
        
        text = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            return_tensors="pt",
        ).to(peft_model.device)
        
        batch_size = inputs['input_ids'].shape[0]
        num_loras = len(LORA_ADAPTERS)
        
        # 偏向 traveling LoRA
        lora_mapping = torch.zeros((batch_size, num_loras), dtype=torch.float32)
        traveling_idx = LORA_ADAPTERS.index("category_lora_traveling")
        lora_mapping[0, traveling_idx] = 0.7
        lora_mapping[0, :] += 0.3 / num_loras
        lora_mapping = lora_mapping.to(peft_model.device)
        
        print(f"\n📊 LoRA Mapping (偏向 traveling):")
        for i, lora_name in enumerate(LORA_ADAPTERS):
            print(f"   - {lora_name}: {lora_mapping[0, i]:.3f}")
        
        # Mixture 模式推理
        print(f"\n🚀 运行 Mixture 模式推理...")
        peft_model.eval()
        with torch.no_grad():
            outputs = peft_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                merging_type='mixture',
                lora_mapping=lora_mapping
            )
        
        print(f"✅ Mixture 模式成功")
        print(f"   输出形状: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Mixture 模式失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🧪 LoRA Fusion/Mixture 功能测试 - Qwen2-VL + Category LoRAs")
    print("=" * 80)
    print(f"\n📂 基础模型: {BASE_MODEL_PATH}")
    print(f"📂 LoRA 目录: {LORA_BASE_PATH}")
    print(f"\n📋 LoRA Adapters ({len(LORA_ADAPTERS)} 个):")
    for lora_name in LORA_ADAPTERS:
        lora_path = os.path.join(LORA_BASE_PATH, lora_name)
        exists = "✅" if os.path.exists(lora_path) else "❌"
        print(f"   {exists} {lora_name}")
    
    temp_dirs = []
    results = []
    
    try:
        # 测试 1: 加载模型和 LoRAs
        peft_model, processor, temp_dirs, success = test_load_model_and_loras()
        results.append(("加载模型和 LoRAs", success))
        
        if not success:
            print("\n❌ 模型加载失败，跳过后续测试")
            return 1
        
        # 测试 2: Fusion 模式
        results.append(("Fusion 模式", test_fusion_mode(peft_model, processor)))
        
        # 测试 3: Mixture 模式
        results.append(("Mixture 模式", test_mixture_mode(peft_model, processor)))
        
    finally:
        # 清理临时目录
        print(f"\n🧹 清理临时文件...")
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"   删除: {temp_dir}")
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:25s}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
