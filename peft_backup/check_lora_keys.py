from safetensors import safe_open

with safe_open('/home/hmpiao/hmpiao/xuerong/FedMABench/lora_category/category_lora_entertainment/adapter_model.safetensors', framework='pt', device='cpu') as f:
    keys = list(f.keys())
    print(f'LoRA 权重总数: {len(keys)}')
    print(f'\nLoRA 权重键名（前30个）:')
    for k in keys[:30]:
        print(f'  {k}')
