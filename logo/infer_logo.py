"""
LOGO Inference Script for InternVL2

Main script for running inference with LOGO (LoRA on the Go) method.
Dynamically selects and merges LoRAs based on input signals.
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

from logo.logo_engine import LOGOEngine, load_lora_configs

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
                        choices=['norm', 'entropy'],
                        help='Signal computation method')
    
    # Generation settings
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


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
    output_path = os.path.join(args.output_dir, f'logo_results_{time_str}.jsonl')
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    test_data = load_jsonl(args.test_data)
    logger.info(f"Loaded {len(test_data)} samples")
    
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
        signal_type=args.signal_type
    )
    
    # Load all LoRAs
    logger.info("Loading all LoRA adapters...")
    engine.load_all_loras()
    
    # Run inference
    logger.info("Starting LOGO inference...")
    results = []
    
    for sample in tqdm(test_data, desc="LOGO Inference"):
        query = sample['query']
        images = sample.get('images', [])
        label = sample.get('response', '')
        
        try:
            # TODO: Implement full signal extraction with model forward pass
            # For now, use a simplified approach: activate all LoRAs and use
            # uniform weights as baseline, then implement proper signal extraction
            
            # Simplified version: just use top-k with uniform weights
            # Full implementation would compute signals from forward pass
            uniform_signals = {name: 1.0 for name in engine.adapter_names}
            selected_loras, weights = engine.select_and_merge(uniform_signals)
            
            # Generate response
            response, _ = inference(
                engine.model,
                template,
                query,
                history=[],
                system=None,
                images=images,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            
            result = {
                'query': query,
                'response': response,
                'label': label,
                'selected_loras': selected_loras,
                'weights': weights,
                'images': images
            }
            
            # Reset for next sample
            engine.reset_merged_adapter()
            
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            result = {
                'query': query,
                'response': f"ERROR: {str(e)}",
                'label': label,
                'selected_loras': [],
                'weights': [],
                'images': images
            }
        
        results.append(result)
        append_to_jsonl(output_path, result)
    
    logger.info(f"Inference complete. Results saved to: {output_path}")
    logger.info(f"Processed {len(results)} samples")
    
    return results


if __name__ == '__main__':
    args = parse_args()
    run_logo_inference(args)
