"""
Test script for LOGO signal extraction.

Verifies that signal extraction works correctly with the InternVL2 model.
"""

import os
import sys
import json

# Set GPU before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from swift.tuners import Swift
from swift.llm.utils import get_model_tokenizer, get_template

from logo.signal_extractor import (
    LOGOSignalExtractor,
    compute_signal,
    normalize_weights,
    select_top_k,
    compute_uniform_signals
)


def test_signal_extraction():
    """Test signal extraction with real model and LoRAs."""
    
    print("=" * 60)
    print("Testing LOGO Signal Extraction")
    print("=" * 60)
    
    # Load base model
    print("\n1. Loading base model...")
    model_kwargs = {
        'device_map': 'auto',
        'low_cpu_mem_usage': True,
    }
    
    model, tokenizer = get_model_tokenizer(
        'internvl2-2b',
        torch.float16,
        model_kwargs,
        model_id_or_path='/home/hmpiao/hmpiao/InternVL2-2B-ModelScope/OpenGVLab/InternVL2-2B'
    )
    print(f"   Base model loaded: {type(model).__name__}")
    
    # Get template
    template = get_template(
        'internvl2',
        tokenizer,
        None,
        2048,
        'truncation_left',
        model=model
    )
    
    # Load LoRA configs
    print("\n2. Loading LoRA configs...")
    configs = []
    for config_path in ['config/app_loras_config_internvl2.json', 
                        'config/category_loras_config_internvl2.json']:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configs.extend(json.load(f))
    print(f"   Found {len(configs)} LoRA configs")
    
    # Load first 3 LoRAs for testing
    test_configs = configs[:3]
    loaded_adapters = []
    
    print("\n3. Loading LoRAs...")
    for cfg in test_configs:
        lora_name = cfg['lora_name']
        lora_path = cfg['lora_path']
        
        if not os.path.exists(lora_path):
            print(f"   [SKIP] {lora_name}: path not found")
            continue
            
        try:
            model = Swift.from_pretrained(
                model,
                lora_path,
                adapter_name=lora_name,
                inference_mode=True
            )
            loaded_adapters.append(lora_name)
            print(f"   [OK] Loaded: {lora_name}")
        except Exception as e:
            print(f"   [FAIL] {lora_name}: {e}")
    
    if len(loaded_adapters) < 2:
        print("\n   [ERROR] Need at least 2 adapters for testing")
        return False
    
    # Update template model
    template.model = model
    
    # Initialize signal extractor (with baseline calibration enabled)
    print("\n4. Initializing signal extractor...")
    extractor = LOGOSignalExtractor(
        model=model,
        adapter_names=loaded_adapters,
        signal_type='norm',
        target_block_idx=-1,
        token_position='last',
        use_baseline_calibration=True  # Enable calibration
    )
    print(f"   Initialized with {len(loaded_adapters)} adapters")
    print(f"   Baseline calibration: enabled")
    
    # Find LoRA layers
    print("\n5. Finding LoRA layers...")
    lora_layers = extractor._find_lora_layers()
    print(f"   Found {len(lora_layers)} LoRA layers total")
    
    target_layers = extractor._find_target_block_lora_layers()
    print(f"   Found {len(target_layers)} LoRA layers in target block")
    
    if target_layers:
        print(f"   Target layers: {list(target_layers.keys())[:3]}...")
    
    # Test with sample input
    print("\n6. Testing signal extraction with sample input...")
    
    # Create simple text input
    test_text = "What is shown in this image?"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    print(f"   Input: '{test_text}'")
    print(f"   Input IDs shape: {input_ids.shape}")
    
    # ========== Compute baseline signals using new API ==========
    print("\n6.1 Computing baseline signals (using compute_baseline API)...")
    
    # Use the built-in baseline computation
    baseline_signals = extractor.compute_baseline(tokenizer=tokenizer)
    
    print(f"   Baseline signals computed:")
    for name in sorted(baseline_signals.keys()):
        print(f"     {name}: {baseline_signals[name]:.4f}")
    
    # ========== Test with brand-specific queries (calibrated signals) ==========
    print("\n6.2 Testing with brand-specific queries (calibrated signals)...")
    print("   (extractor automatically applies ratio-based calibration)")
    
    brand_queries = [
        ("adidas", "I want to buy Adidas running shoes online"),
        ("amazon", "Search for products on Amazon marketplace"),
        ("calendar", "Schedule a meeting for next Monday at 3pm"),
    ]
    
    correct_count = 0
    total_count = len(brand_queries)
    
    for expected_brand, query in brand_queries:
        query_inputs = tokenizer(query, return_tensors="pt")
        query_input_ids = query_inputs['input_ids'].to(model.device)
        query_attention_mask = query_inputs['attention_mask'].to(model.device)
        
        # extract_signals now automatically applies calibration
        calibrated_signals = extractor.extract_signals(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask
        )
        
        # Normalize weights
        weights = normalize_weights(calibrated_signals)
        
        # Find top adapter
        top_adapter = max(calibrated_signals, key=calibrated_signals.get)
        is_correct = expected_brand in top_adapter
        if is_correct:
            correct_count += 1
        
        print(f"\n   Query: '{query}'")
        print(f"   Expected: {expected_brand}")
        print(f"   ")
        print(f"   {'Adapter':<20} {'Calibrated Signal':>18} {'Weight':>12}")
        print(f"   {'-'*20} {'-'*18} {'-'*12}")
        for name in sorted(calibrated_signals.keys()):
            marker = " <-- TOP" if name == top_adapter else ""
            print(f"   {name:<20} {calibrated_signals[name]:>18.4f} {weights[name]:>11.2%}{marker}")
        
        result = "✓ CORRECT" if is_correct else "✗ WRONG"
        print(f"\n   Result: {result} (Top: {top_adapter})")
    
    print(f"\n   Summary: {correct_count}/{total_count} queries matched expected adapter")
    
    # Extract signals
    try:
        signals = extractor.extract_signals(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        print(f"\n   Extracted signals:")
        for name, score in signals.items():
            print(f"     {name}: {score:.6f}")
        
        # Test different signal types
        print("\n7. Testing different signal types...")
        
        # Entropy-based
        extractor_entropy = LOGOSignalExtractor(
            model=model,
            adapter_names=loaded_adapters,
            signal_type='entropy',
            target_block_idx=-1,
            token_position='last'
        )
        
        signals_entropy = extractor_entropy.extract_signals(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        print(f"   Entropy-based signals:")
        for name, score in signals_entropy.items():
            print(f"     {name}: {score:.6f}")
        
        # Test normalization
        print("\n8. Testing weight normalization...")
        weights = normalize_weights(signals)
        print(f"   Normalized weights (sum={sum(weights.values()):.4f}):")
        for name, weight in weights.items():
            print(f"     {name}: {weight:.4f}")
        
        # Test top-k selection
        print("\n9. Testing top-k selection...")
        top_k = select_top_k(signals, k=2)
        print(f"   Top-2 adapters: {top_k}")
        
        # Test uniform signals
        print("\n10. Testing uniform signals...")
        uniform = compute_uniform_signals(loaded_adapters)
        print(f"   Uniform signals: {uniform}")
        
        print("\n" + "=" * 60)
        print("All Signal Extraction Tests Passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n   [ERROR] Signal extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_with_image():
    """Test signal extraction with image input."""
    
    print("\n" + "=" * 60)
    print("Testing LOGO Signal Extraction with Image")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
    model_kwargs = {
        'device_map': 'auto',
        'low_cpu_mem_usage': True,
    }
    
    model, tokenizer = get_model_tokenizer(
        'internvl2-2b',
        torch.float16,
        model_kwargs,
        model_id_or_path='/home/hmpiao/hmpiao/InternVL2-2B-ModelScope/OpenGVLab/InternVL2-2B'
    )
    
    # Get template
    template = get_template(
        'internvl2',
        tokenizer,
        None,
        2048,
        'truncation_left',
        model=model
    )
    
    # Load LoRAs
    print("\n2. Loading LoRAs...")
    configs = []
    for config_path in ['config/app_loras_config_internvl2.json', 
                        'config/category_loras_config_internvl2.json']:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configs.extend(json.load(f))
    
    loaded_adapters = []
    for cfg in configs[:3]:
        lora_path = cfg['lora_path']
        lora_name = cfg['lora_name']
        
        if os.path.exists(lora_path):
            try:
                model = Swift.from_pretrained(
                    model,
                    lora_path,
                    adapter_name=lora_name,
                    inference_mode=True
                )
                loaded_adapters.append(lora_name)
                print(f"   [OK] Loaded: {lora_name}")
            except Exception as e:
                print(f"   [FAIL] {lora_name}: {e}")
    
    if len(loaded_adapters) < 2:
        print("   [SKIP] Not enough adapters loaded")
        return True
    
    # Update template
    template.model = model
    
    # Load sample with image
    print("\n3. Loading sample with image...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if os.path.exists('data/Val_100.jsonl'):
        with open('data/Val_100.jsonl', 'r') as f:
            sample = json.loads(f.readline().strip())
        
        query = sample['query']
        images = sample.get('images', [])
        
        # Resolve image paths
        resolved_images = []
        for img_path in images[:1]:  # Use only first image
            if img_path.startswith('./../'):
                img_path = img_path.replace('./../', '')
                img_path = os.path.join(os.path.dirname(project_root), img_path)
            if os.path.exists(img_path):
                resolved_images.append(img_path)
        
        if resolved_images:
            print(f"   Found image: {resolved_images[0]}")
            
            # Prepare query
            query_lines = query.split('\n')
            non_image_lines = [l for l in query_lines if l != '<image>']
            new_query = '\n'.join(['<image>'] * len(resolved_images) + non_image_lines)
            
            # Encode
            example = {
                'query': new_query,
                'history': [],
                'system': None,
                'images': resolved_images,
                'audios': [],
                'videos': [],
                'tools': None,
                'objects': None,
            }
            
            print("\n4. Encoding input with image...")
            inputs, _ = template.encode(example)
            
            print(f"   Input keys: {list(inputs.keys())}")
            if 'inputs_embeds' in inputs:
                print(f"   inputs_embeds shape: {inputs['inputs_embeds'].shape}")
            
            # Extract signals
            print("\n5. Extracting signals with image input...")
            extractor = LOGOSignalExtractor(
                model=model,
                adapter_names=loaded_adapters,
                signal_type='norm',
                target_block_idx=-1,
                token_position='last'
            )
            
            signal_inputs = {}
            if 'inputs_embeds' in inputs:
                signal_inputs['inputs_embeds'] = inputs['inputs_embeds'].unsqueeze(0).to(model.device)
            if 'attention_mask' in inputs:
                signal_inputs['attention_mask'] = torch.tensor([inputs['attention_mask']]).to(model.device)
            
            signals = extractor.extract_signals(**signal_inputs)
            
            # Normalize weights
            weights = normalize_weights(signals)
            
            # Find top adapter
            top_adapter = max(signals, key=signals.get)
            
            print(f"\n   Signals with image input:")
            for name in sorted(signals.keys()):
                marker = " <-- TOP" if name == top_adapter else ""
                print(f"     {name}: signal={signals[name]:.6f}, weight={weights[name]:.2%}{marker}")
            
            print(f"\n   Top adapter for this image: {top_adapter}")
            
            print("\n" + "=" * 60)
            print("Image Signal Extraction Test Passed!")
            print("=" * 60)
            return True
        else:
            print("   [SKIP] No valid images found")
            return True
    else:
        print("   [SKIP] Test data not found")
        return True


if __name__ == '__main__':
    success1 = test_signal_extraction()
    success2 = test_signal_with_image()
    
    if success1 and success2:
        print("\n\nAll tests passed!")
        sys.exit(0)
    else:
        print("\n\nSome tests failed!")
        sys.exit(1)
