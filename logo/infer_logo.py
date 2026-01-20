"""
LOGO Inference Script for InternVL2

Main script for running inference with LOGO (LoRA on the Go) method.
Dynamically selects and merges LoRAs based on input signals.

Usage:
    python logo/infer_logo.py --test_data data/Val_100.jsonl --signal_type norm --top_k 5
"""

import os
import sys
import json
import argparse
import datetime as dt
from typing import Dict, List, Any, Optional
from tqdm import tqdm

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swift.utils import get_logger, append_to_jsonl, seed_everything
from swift.llm.utils import (
    get_model_tokenizer, 
    get_template, 
    inference,
    InferArguments
)

from logo.logo_engine import LOGOEngine, load_lora_configs, create_logo_engine

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description='LOGO Inference for InternVL2')
    
    # Data paths
    parser.add_argument('--test_data', type=str, default='data/Val_100.jsonl',
                        help='Path to test dataset (JSONL format)')
    parser.add_argument('--app_config', type=str, 
                        default='config/app_loras_config_internvl2.json',
                        help='Path to app-level LoRA config')
    parser.add_argument('--category_config', type=str,
                        default='config/category_loras_config_internvl2.json',
                        help='Path to category-level LoRA config')
    parser.add_argument('--output_dir', type=str, default='output/logo_results',
                        help='Output directory for results')
    
    # Model settings
    parser.add_argument('--model_type', type=str, default='internvl2-2b',
                        help='Model type')
    parser.add_argument('--model_path', type=str,
                        default='/home/hmpiao/hmpiao/InternVL2-2B-ModelScope/OpenGVLab/InternVL2-2B',
                        help='Path to base model')
    
    # LOGO settings
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top LoRAs to select')
    parser.add_argument('--signal_type', type=str, default='norm',
                        choices=['norm', 'entropy', 'embedding', 'uniform'],
                        help='Signal computation method')
    parser.add_argument('--target_block_idx', type=int, default=-1,
                        help='Transformer block for signal extraction (-1 = last)')
    parser.add_argument('--token_position', type=str, default='last',
                        choices=['last', 'first', 'mean'],
                        help='Token position for signal computation')
    parser.add_argument('--combination_type', type=str, default='linear',
                        choices=['linear', 'svd', 'cat'],
                        help='LoRA merging method')
    
    # Generation settings
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    
    # Debug settings
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (process fewer samples)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (for debugging)')
    
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def resolve_image_paths(images: List[str], project_root: str) -> List[str]:
    """Resolve relative image paths to absolute paths."""
    resolved = []
    for img_path in images:
        if img_path.startswith('./../'):
            # Relative path from data directory
            img_path = img_path.replace('./../', '')
            img_path = os.path.join(os.path.dirname(project_root), img_path)
        elif not os.path.isabs(img_path):
            # Relative path from project root
            img_path = os.path.join(project_root, img_path)
        
        if os.path.exists(img_path):
            resolved.append(img_path)
        else:
            logger.warning(f"Image not found: {img_path}")
    
    return resolved


def prepare_query(query: str, num_images: int) -> str:
    """Adjust query to match number of available images."""
    query_lines = query.split('\n')
    non_image_lines = [l for l in query_lines if l != '<image>']
    return '\n'.join(['<image>'] * num_images + non_image_lines)


def prepare_model(args) -> tuple:
    """Load and prepare the base InternVL2 model."""
    logger.info(f"Loading base model: {args.model_type}")
    
    model_kwargs = {
        'device_map': 'auto',
        'low_cpu_mem_usage': True,
    }
    
    model, tokenizer = get_model_tokenizer(
        args.model_type,
        torch.float16,
        model_kwargs,
        model_id_or_path=args.model_path
    )
    
    # Get template
    template = get_template(
        'internvl2',
        tokenizer,
        None,  # system prompt
        2048,  # max_length
        'truncation_left',
        model=model
    )
    
    return model, tokenizer, template


def run_logo_inference(args):
    """Main LOGO inference function."""
    seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    time_str = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    output_path = os.path.join(
        args.output_dir, 
        f'logo_results_{args.signal_type}_k{args.top_k}_{time_str}.jsonl'
    )
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    test_data = load_jsonl(args.test_data)
    logger.info(f"Loaded {len(test_data)} samples")
    
    # Limit samples for debugging
    if args.debug or args.num_samples:
        num_samples = args.num_samples or 5
        test_data = test_data[:num_samples]
        logger.info(f"Debug mode: processing {len(test_data)} samples")
    
    # Load LoRA configs
    config_paths = []
    if os.path.exists(args.app_config):
        config_paths.append(args.app_config)
    if os.path.exists(args.category_config):
        config_paths.append(args.category_config)
    
    lora_configs = load_lora_configs(config_paths)
    
    if not lora_configs:
        logger.error("No LoRA configs found!")
        return
    
    # Load base model
    model, tokenizer, template = prepare_model(args)
    
    # Initialize LOGO engine
    logger.info("Initializing LOGO engine...")
    engine = LOGOEngine(
        base_model=model,
        lora_configs=lora_configs,
        top_k=args.top_k,
        signal_type=args.signal_type,
        target_block_idx=args.target_block_idx,
        token_position=args.token_position,
        combination_type=args.combination_type
    )
    
    # Load all LoRAs
    logger.info("Loading all LoRA adapters...")
    engine.load_all_loras()
    
    # Update template model reference
    template.model = engine.model
    
    # Compute baseline signals for calibration (if using norm/entropy signal type)
    if args.signal_type in ['norm', 'entropy']:
        logger.info("Computing baseline signals for calibration...")
        engine.set_tokenizer(tokenizer)
        engine.compute_baseline(tokenizer=tokenizer)
    
    # Print engine info
    logger.info(f"LOGO Engine Info: {engine.get_adapter_info()}")
    
    # Project root for resolving paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Run inference
    logger.info("Starting LOGO inference...")
    results = []
    
    for idx, sample in enumerate(tqdm(test_data, desc="LOGO Inference")):
        query = sample.get('query', '')
        images = sample.get('images', [])
        label = sample.get('response', '')
        
        try:
            # Resolve image paths
            resolved_images = resolve_image_paths(images, project_root)
            
            # Adjust query for available images
            adjusted_query = prepare_query(query, len(resolved_images))
            
            # Prepare example for template encoding
            example = {
                'query': adjusted_query,
                'history': [],
                'system': None,
                'images': resolved_images,
                'audios': [],
                'videos': [],
                'tools': None,
                'objects': None,
            }
            
            # Encode to get inputs (for signal extraction)
            inputs, tokenizer_kwargs = template.encode(example)
            
            # Extract signals and merge adapters
            # Prepare inputs for signal extraction
            signal_inputs = {}
            if 'input_ids' in inputs:
                signal_inputs['input_ids'] = torch.tensor([inputs['input_ids']]).to(engine.model.device)
            if 'attention_mask' in inputs:
                signal_inputs['attention_mask'] = torch.tensor([inputs['attention_mask']]).to(engine.model.device)
            if 'inputs_embeds' in inputs:
                signal_inputs['inputs_embeds'] = inputs['inputs_embeds'].unsqueeze(0).to(engine.model.device)
            if 'pixel_values' in inputs:
                signal_inputs['pixel_values'] = inputs['pixel_values'].to(engine.model.device)
            
            # LOGO: Extract signals and merge adapters
            selected_loras, weights = engine.process_input(**signal_inputs)
            
            # Update template model with merged adapter
            template.model = engine.model
            
            # Generate response using merged adapter
            if resolved_images:
                response, _ = inference(
                    engine.model,
                    template,
                    adjusted_query,
                    history=[],
                    system=None,
                    images=resolved_images,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature
                )
            else:
                # Text-only inference
                response, _ = inference(
                    engine.model,
                    template,
                    adjusted_query,
                    history=[],
                    system=None,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature
                )
            
            result = {
                'idx': idx,
                'query': query,
                'response': response,
                'label': label,
                'selected_loras': selected_loras,
                'weights': weights,
                'num_images': len(resolved_images),
                'signal_type': args.signal_type
            }
            
            if args.debug:
                logger.info(f"\nSample {idx}:")
                logger.info(f"  Query: {query[:100]}...")
                logger.info(f"  Selected LoRAs: {list(zip(selected_loras, [f'{w:.3f}' for w in weights]))}")
                logger.info(f"  Response: {response[:200]}...")
            
            # Reset for next sample
            engine.reset_merged_adapter()
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'idx': idx,
                'query': query,
                'response': f"ERROR: {str(e)}",
                'label': label,
                'selected_loras': [],
                'weights': [],
                'num_images': len(images),
                'signal_type': args.signal_type
            }
        
        results.append(result)
        append_to_jsonl(output_path, result)
        
        # Periodically clean up merged adapters to save memory
        if (idx + 1) % 100 == 0:
            engine.delete_merged_adapters()
    
    # Summary
    successful = sum(1 for r in results if not r['response'].startswith('ERROR'))
    logger.info(f"\n{'='*60}")
    logger.info(f"Inference complete!")
    logger.info(f"  Total samples: {len(results)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {len(results) - successful}")
    logger.info(f"  Results saved to: {output_path}")
    logger.info(f"{'='*60}")
    
    return results


if __name__ == '__main__':
    args = parse_args()
    run_logo_inference(args)
