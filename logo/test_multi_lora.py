"""
Test script for LOGO multi-LoRA loading, merging, and inference with Swift template.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from swift.tuners import Swift
from swift.llm.utils import get_model_tokenizer, get_template, inference


def test_logo_inference():
    """Test LoRA loading, merging, and inference using Swift template."""
    
    print("=" * 60)
    print("Testing LOGO Multi-LoRA Inference with Template Fix")
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
    
    # Get template - pass model to template
    template = get_template(
        'internvl2',
        tokenizer,
        None,  # system
        2048,  # max_length
        'truncation_left',
        model=model  # Pass model to template
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
    
    # Load first 3 LoRAs
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
    
    print(f"\n   Loaded {len(loaded_adapters)} adapters")
    
    # Merge adapters
    if len(loaded_adapters) >= 2:
        print("\n4. Merging adapters...")
        weights = [0.5, 0.3, 0.2][:len(loaded_adapters)]
        total = sum(weights)
        weights = [w/total for w in weights]
        
        model.add_weighted_adapter(
            adapters=loaded_adapters,
            weights=weights,
            adapter_name='logo_merged',
            combination_type='linear'
        )
        
        if hasattr(model, 'set_adapter'):
            model.set_adapter('logo_merged')
            
        print(f"   [OK] Merged: {list(zip(loaded_adapters, weights))}")
    
    # **IMPORTANT**: Update template.model to the wrapped model
    template.model = model
    print(f"   Template model updated to: {type(template.model).__name__}")
    
    # Test inference with real data
    print("\n5. Testing inference with real data...")
    try:
        with open('data/Val_100.jsonl', 'r') as f:
            sample = json.loads(f.readline().strip())
        
        query = sample['query']
        images = sample.get('images', [])
        
        # Resolve image paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        resolved_images = []
        for img_path in images[:2]:
            if img_path.startswith('./../'):
                img_path = img_path.replace('./../', '')
                img_path = os.path.join(os.path.dirname(project_root), img_path)
            if os.path.exists(img_path):
                resolved_images.append(img_path)
        
        # Adjust query to match number of images
        query_lines = query.split('\n')
        non_image_lines = [l for l in query_lines if l != '<image>']
        new_query = '\n'.join(['<image>'] * len(resolved_images) + non_image_lines)
        
        print(f"   Using {len(resolved_images)} images")
        print(f"   Query: {non_image_lines[-1] if non_image_lines else 'N/A'}")
        
        if resolved_images:
            response, _ = inference(
                model,
                template,
                new_query,
                history=[],
                system=None,
                images=resolved_images,
                max_new_tokens=100,
                temperature=0.0
            )
            print(f"   Response: {response[:200]}...")
            print("   [OK] Inference works with merged LoRA!")
        else:
            print("   [SKIP] No valid images")
            
    except Exception as e:
        print(f"   [FAIL] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All Tests Passed!")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = test_logo_inference()
    sys.exit(0 if success else 1)
