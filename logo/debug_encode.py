"""Debug script to understand why template.encode returns empty inputs for PeftModel."""

import os
import sys
import json

# 指定使用的 GPU（避免 OOM）
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from swift.tuners import Swift
from swift.llm.utils import get_model_tokenizer, get_template

def debug_encode():
    print("=" * 60)
    print("Debugging template.encode for PeftModel")
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
    print(f"   Base model type: {type(model).__name__}")
    print(f"   Base model dtype: {model.dtype}")
    
    # Get template - pass model to template
    template = get_template(
        'internvl2',
        tokenizer,
        None,  # system
        2048,  # max_length
        'truncation_left',
        model=model  # Pass model to template
    )
    print(f"   Template model type: {type(template.model).__name__}")
    
    # Test encode BEFORE loading LoRA
    print("\n2. Testing encode BEFORE loading LoRA...")
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
    
    example_before = {
        'query': new_query,
        'history': [],
        'system': None,
        'images': resolved_images,
        'audios': [],
        'videos': [],
        'tools': None,
        'objects': None,
    }
    
    inputs_before, tokenizer_kwargs_before = template.encode(example_before)
    print(f"   inputs_before keys: {list(inputs_before.keys())}")
    print(f"   input_ids length: {len(inputs_before.get('input_ids', []))}")
    if 'inputs_embeds' in inputs_before:
        print(f"   inputs_embeds shape: {inputs_before['inputs_embeds'].shape}")
    
    # Now load LoRA
    print("\n3. Loading LoRA...")
    lora_path = '/home/hmpiao/hmpiao/jinyike/FedMABench/lora_category_internvl2-2b/category_lora_Entertainment_internvl2-2b/internvl2-2b/v4-20260119-232917/global_lora_2'
    model = Swift.from_pretrained(
        model,
        lora_path,
        adapter_name='app_lora_adidas',
        inference_mode=True
    )
    print(f"   Model type after LoRA: {type(model).__name__}")
    
    # Update template model
    template.model = model
    print(f"   Template model type updated: {type(template.model).__name__}")
    
    # Test encode AFTER loading LoRA
    print("\n4. Testing encode AFTER loading LoRA...")
    example_after = {
        'query': new_query,
        'history': [],
        'system': None,
        'images': resolved_images,
        'audios': [],
        'videos': [],
        'tools': None,
        'objects': None,
    }
    
    try:
        inputs_after, tokenizer_kwargs_after = template.encode(example_after)
        print(f"   inputs_after keys: {list(inputs_after.keys())}")
        print(f"   input_ids length: {len(inputs_after.get('input_ids', []))}")
        if 'inputs_embeds' in inputs_after:
            print(f"   inputs_embeds shape: {inputs_after['inputs_embeds'].shape}")
    except Exception as e:
        print(f"   [ERROR] encode failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    if inputs_after:
        print("SUCCESS: encode returns non-empty inputs with PeftModel")
    else:
        print("FAILURE: encode returns empty inputs with PeftModel")
    print("=" * 60)

if __name__ == '__main__':
    debug_encode()
