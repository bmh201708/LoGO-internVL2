#!/usr/bin/env python3
"""
测试 Mixture 模式输出是否正确。

比较：
1. 单个 LoRA 的输出（正常模式）
2. Mixture 模式（权重为 1.0 的单个 LoRA）的输出

如果两者相同，说明 Mixture 实现正确；如果不同，说明有 bug。
"""

import os
import sys
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from swift.llm.utils import get_model_tokenizer, get_template, inference
from swift.tuners.lora_layers import logo_mixture_context, Linear
from swift.tuners import Swift


def reset_debug_counters():
    """重置调试计数器"""
    Linear._debug_mixture_call_count = 0
    Linear._debug_normal_call_count = 0


def print_debug_counters(test_name):
    """打印调试计数器"""
    print(f"   [DEBUG] {test_name}: mixture_calls={Linear._debug_mixture_call_count}, normal_calls={Linear._debug_normal_call_count}")


def main():
    print("=" * 60)
    print("Testing Mixture Mode Output Correctness")
    print("=" * 60)
    
    # 1. Load base model
    print("\n1. Loading base model...")
    model_kwargs = {
        'device_map': 'auto',
        'low_cpu_mem_usage': True,
    }
    model, tokenizer = get_model_tokenizer(
        'internvl2-2b',
        torch.float16,
        model_kwargs,
        model_dir='/home/hmpiao/hmpiao/InternVL2-2B-ModelScope/OpenGVLab/InternVL2-2B'
    )
    
    # 2. Load a single LoRA using Swift
    print("\n2. Loading a single LoRA using Swift...")
    lora_path = "/home/hmpiao/hmpiao/jinyike/FedMABench/lora_category_internvl2-2b/category_lora_Entertainment_internvl2-2b/internvl2-2b/v3-20260118-135112/global_lora_8"
    
    model = Swift.from_pretrained(model, lora_path, adapter_name="test_lora")
    print(f"   Loaded adapter: test_lora")
    
    # Check active adapters
    if hasattr(model, 'active_adapters'):
        print(f"   Active adapters: {model.active_adapters}")
    if hasattr(model, 'peft_config'):
        print(f"   Peft config keys: {list(model.peft_config.keys())}")
    
    # Get template
    template = get_template('internvl2', tokenizer, None, 2048, 'truncation_left', model=model)
    
    # Test query
    test_query = "Play a YouTube video"
    
    print(f"\n3. Test query: '{test_query}'")
    
    # ==================== Test 1: Normal single LoRA mode ====================
    print("\n" + "=" * 60)
    print("TEST 1: Normal single LoRA mode")
    print("=" * 60)
    
    # Make sure context is reset
    logo_mixture_context.reset()
    reset_debug_counters()
    
    response1, _ = inference(
        model,
        template,
        test_query,
        history=[],
        max_new_tokens=100,
        temperature=0.0,
    )
    print(f"Response: {response1}")
    print_debug_counters("Test1")
    
    # ==================== Test 2: Mixture mode with weight=1.0 ====================
    print("\n" + "=" * 60)
    print("TEST 2: Mixture mode with weight=1.0 for the same LoRA")
    print("=" * 60)
    
    # Create lora_mapping with weight=1.0 for the single adapter
    adapter_names = ["test_lora"]
    lora_mapping = torch.tensor([[1.0]], device=model.device)  # (batch=1, num_adapters=1)
    
    reset_debug_counters()
    response2, _ = inference(
        model,
        template,
        test_query,
        history=[],
        max_new_tokens=100,
        temperature=0.0,
        merging_type='mixture',
        lora_mapping=lora_mapping,
        mixture_adapter_names=adapter_names
    )
    print(f"Response: {response2}")
    print_debug_counters("Test2")
    
    # ==================== Test 3: Mixture mode with weight=0.5 ====================
    print("\n" + "=" * 60)
    print("TEST 3: Mixture mode with weight=0.5 (should be different)")
    print("=" * 60)
    
    lora_mapping_half = torch.tensor([[0.5]], device=model.device)
    
    reset_debug_counters()
    response3, _ = inference(
        model,
        template,
        test_query,
        history=[],
        max_new_tokens=100,
        temperature=0.0,
        merging_type='mixture',
        lora_mapping=lora_mapping_half,
        mixture_adapter_names=adapter_names
    )
    print(f"Response: {response3}")
    print_debug_counters("Test3")
    
    # ==================== Test 4: Mixture mode with weight=0.0 (CRITICAL TEST) ====================
    print("\n" + "=" * 60)
    print("TEST 4: Mixture mode with weight=0.0 (should be BASE MODEL output)")
    print("=" * 60)
    
    lora_mapping_zero = torch.tensor([[0.0]], device=model.device)
    
    reset_debug_counters()
    response4, _ = inference(
        model,
        template,
        test_query,
        history=[],
        max_new_tokens=100,
        temperature=0.0,
        merging_type='mixture',
        lora_mapping=lora_mapping_zero,
        mixture_adapter_names=adapter_names
    )
    print(f"Response: {response4}")
    print_debug_counters("Test4")
    
    # ==================== Test 5: No LoRA (disable adapter) for comparison ====================
    print("\n" + "=" * 60)
    print("TEST 5: Disable LoRA (pure base model)")
    print("=" * 60)
    
    # Disable all adapters
    logo_mixture_context.reset()
    if hasattr(model, 'disable_adapter_layers'):
        model.disable_adapter_layers()
    
    response5, _ = inference(
        model,
        template,
        test_query,
        history=[],
        max_new_tokens=100,
        temperature=0.0,
    )
    print(f"Response: {response5}")
    
    # Re-enable adapters
    if hasattr(model, 'enable_adapter_layers'):
        model.enable_adapter_layers()
    
    # ==================== Comparison ====================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Test 1 (normal mode) == Test 2 (mixture weight=1.0): {response1 == response2}")
    print(f"Test 1 (normal mode) == Test 3 (mixture weight=0.5): {response1 == response3}")
    print(f"Test 1 (normal mode) == Test 4 (mixture weight=0.0): {response1 == response4}")
    print(f"Test 4 (mixture weight=0.0) == Test 5 (pure base): {response4 == response5}")
    
    print("\n" + "-" * 60)
    print("ANALYSIS:")
    print("-" * 60)
    
    # Check if mixture weight=1.0 equals normal mode
    if response1 == response2:
        print("✓ [PASS] Mixture weight=1.0 == Normal mode")
    else:
        print("✗ [FAIL] Mixture weight=1.0 != Normal mode (BUG!)")
        print(f"  Normal: {response1[:100]}...")
        print(f"  Mix1.0: {response2[:100]}...")
    
    # Check if mixture weight=0.0 equals pure base model
    if response4 == response5:
        print("✓ [PASS] Mixture weight=0.0 == Pure base model")
    else:
        print("✗ [FAIL] Mixture weight=0.0 != Pure base model (BUG!)")
        print(f"  Mix0.0: {response4[:100]}...")
        print(f"  Base:   {response5[:100]}...")
    
    # Check if weight affects output
    if response1 != response4:
        print("✓ [PASS] LoRA weight affects output (weight=1.0 != weight=0.0)")
    else:
        print("⚠ [WARN] LoRA has no effect! (weight=1.0 == weight=0.0)")
        print("  This could mean: 1) LoRA is not applied, or 2) Query is too simple")
    
    # Check if intermediate weight produces different output
    if response3 != response1 and response3 != response4:
        print("✓ [PASS] Intermediate weight (0.5) produces different output")
    elif response3 == response1:
        print("⚠ [WARN] weight=0.5 produces SAME output as weight=1.0")
        print("  Possible: 1) argmax sampling masks the difference, or 2) Bug in mixture")
    elif response3 == response4:
        print("⚠ [WARN] weight=0.5 produces SAME output as weight=0.0")


if __name__ == "__main__":
    main()
