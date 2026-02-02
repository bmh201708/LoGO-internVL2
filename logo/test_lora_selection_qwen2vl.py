#!/usr/bin/env python
"""
测试 Qwen2-VL-7B 的 LoRA 选择准确性

验证 LOGO 方法能否根据输入正确选择相关的 LoRA
"""

import os
import sys
import json

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['MAX_PIXELS'] = '100000'
os.environ['MAX_NUM'] = '6'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from swift.tuners import Swift
from swift.llm.utils import get_model_tokenizer, get_template

from logo.logo_engine import LOGOEngine, load_lora_configs, LoRAConfig
from logo.signal_extractor import normalize_weights, select_top_k


def test_lora_selection():
    """测试不同 App 查询的 LoRA 选择"""
    
    print("=" * 70)
    print("Qwen2-VL-7B LoRA Selection Test")
    print("=" * 70)
    
    # 测试用例：每个 App 一个典型查询
    test_cases = [
        # (query, expected_app_lora)
        ("In the Amazon app, search for running shoes", "app_lora_amazon_qwen2vl"),
        ("In the YouTube app, search for music videos", "app_lora_youtube_qwen2vl"),
        ("In the Gmail app, compose a new email", "app_lora_gmail_qwen2vl"),
        ("In the Calendar app, create a new event", "app_lora_calendar_qwen2vl"),
        ("In the Clock app, set an alarm for 7am", "app_lora_clock_qwen2vl"),
        ("In the Google Maps app, find directions to the airport", "app_lora_google_maps_qwen2vl"),
        ("In the eBay app, search for vintage watches", "app_lora_ebay_qwen2vl"),
        ("In the Reminder app, set a reminder for tomorrow", "app_lora_reminder_qwen2vl"),
    ]
    
    # 加载模型
    print("\n[1/3] Loading Qwen2-VL-7B model...")
    model_type = 'qwen2-vl-7b-instruct'
    model_path = '/home/hmpiao/hmpiao/Qwen2-VL-7B-Instruct'
    
    model_kwargs = {
        'device_map': 'auto',
        'low_cpu_mem_usage': True,
    }
    
    model, tokenizer = get_model_tokenizer(
        model_type,
        torch.float16,
        model_kwargs,
        model_id_or_path=model_path
    )
    
    # 加载 LoRA 配置
    print("\n[2/3] Loading App LoRAs...")
    config_path = 'config/app_loras_config_qwen2vl.json'
    lora_configs = load_lora_configs([config_path])
    print(f"  Loaded {len(lora_configs)} LoRA configs")
    
    # 创建 LOGO Engine
    engine = LOGOEngine(
        base_model=model,
        lora_configs=lora_configs,
        top_k=3,
        signal_type='norm',
        target_block_idx=-1,  # 最后一层
        token_position='last',
        use_baseline_calibration=False  # 禁用 baseline 以便看原始信号
    )
    
    # 加载所有 LoRAs
    engine.load_all_loras()
    
    # 测试每个查询
    print("\n[3/3] Testing LoRA selection...")
    print("=" * 70)
    
    correct = 0
    total = len(test_cases)
    
    for query, expected_lora in test_cases:
        print(f"\n>>> Query: {query}")
        print(f"    Expected LoRA: {expected_lora}")
        
        # 编码查询
        inputs = tokenizer(query, return_tensors="pt")
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        
        # 提取信号
        signals = engine.extract_signals(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 选择 top-3
        selected_names = select_top_k(signals, k=3)
        weights = normalize_weights({name: signals[name] for name in selected_names})
        
        # 检查是否选中了期望的 LoRA
        is_correct = expected_lora in selected_names
        rank = selected_names.index(expected_lora) + 1 if is_correct else -1
        
        if is_correct:
            correct += 1
            status = f"✓ Correct (Rank {rank})"
        else:
            status = "✗ Wrong"
        
        print(f"    {status}")
        print(f"    Top-3 Selected LoRAs:")
        for i, name in enumerate(selected_names):
            marker = " <<<" if name == expected_lora else ""
            print(f"      {i+1}. {name}: {weights[name]:.4f}{marker}")
        
        # 显示所有信号 (排序)
        print(f"    All signals (sorted):")
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        for name, sig in sorted_signals[:5]:  # 只显示 top-5
            marker = " <<<" if name == expected_lora else ""
            print(f"      {name}: {sig:.6f}{marker}")
    
    # 总结
    print("\n" + "=" * 70)
    print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("=" * 70)
    
    return correct, total


if __name__ == '__main__':
    test_lora_selection()
