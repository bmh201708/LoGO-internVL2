"""
LOGO Inference Script for InternVL2

Main script for running inference with LOGO (LoRA on the Go) method.
Dynamically selects and merges LoRAs based on input signals.

Usage:
    python logo/infer_logo.py --test_data data/Val_100.jsonl --signal_type norm --top_k 5

python logo/infer_logo.py --model_type qwen2-vl-2b-instruct  --test_data /data0/piaohongming/jinyike/data/data-test/app/amazon_train.jsonl --num_samples 1 --signal_type uniform --merge_method add_weighted_adapter --lora_type app --no_baseline_calibration --output_dir /data0/piaohongming/jinyike/LoGO-internVL2/output/compat_check
"""

import os
import sys

# 设置环境变量以防止 OOM（必须在导入 swift 之前）
if 'MAX_PIXELS' not in os.environ:
    os.environ['MAX_PIXELS'] = '100000'
if 'MAX_NUM' not in os.environ:
    os.environ['MAX_NUM'] = '9'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# 调试输出
print(f"[DEBUG] MAX_PIXELS = {os.environ.get('MAX_PIXELS', 'NOT SET')}")
print(f"[DEBUG] MAX_NUM = {os.environ.get('MAX_NUM', 'NOT SET')}")
print(f"[DEBUG] PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')


# Model path defaults
MODEL_PATHS = {
    'internvl2-2b': '/data0/piaohongming/InternVL2-2B',
    'qwen2-vl-2b-instruct': '/data0/piaohongming/models/Qwen2-VL-2B-Instruct',
    'qwen2-vl-7b-instruct': '/data0/piaohongming/models/Qwen2-VL-7B-Instruct',
}

# Config file paths by model type
CONFIG_PATHS = {
    'internvl2-2b': {
        'app': os.path.join(CONFIG_DIR, 'app_loras_config_internvl2.json'),
        'category': os.path.join(CONFIG_DIR, 'category_loras_config_internvl2.json'),
    },
    'qwen2-vl-2b-instruct': {
        'app': os.path.join(CONFIG_DIR, 'app_loras_config_qwen2b.json'),
        'category': os.path.join(CONFIG_DIR, 'category_loras_config_qwen2vl.json'),
    },
    'qwen2-vl-7b-instruct': {
        'app': os.path.join(CONFIG_DIR, 'app_loras_config_qwen7b.json'),
        'category': os.path.join(CONFIG_DIR, 'category_loras_config_qwen2vl.json'),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='LOGO Inference for VLM (InternVL2 / Qwen2-VL)')
    
    # Model settings (first, as other defaults depend on it)
    parser.add_argument('--model_type', type=str, default='internvl2-2b',
                        choices=['internvl2-2b', 'qwen2-vl-2b-instruct', 'qwen2-vl-7b-instruct'],
                        help='Model type: internvl2-2b, qwen2-vl-2b-instruct, or qwen2-vl-7b-instruct')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to base model (auto-detected if not specified)')
    
    # Data paths
    parser.add_argument('--test_data', type=str, default='data/Val_100.jsonl',
                        help='Path to test dataset (JSONL format)')
    parser.add_argument('--app_config', type=str, default=None,
                        help='Path to app-level LoRA config (auto-detected if not specified)')
    parser.add_argument('--category_config', type=str, default=None,
                        help='Path to category-level LoRA config (auto-detected if not specified)')
    parser.add_argument('--output_dir', type=str, default='output/logo_results',
                        help='Output directory for results')
    
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
                        help='LoRA combination method for add_weighted_adapter')
    parser.add_argument('--merge_method', type=str, default='mixture',
                        choices=['mixture', 'add_weighted_adapter'],
                        help='LoRA merging method: mixture (output-level, 论文推荐) or add_weighted_adapter (parameter-level)')
    parser.add_argument('--no_baseline_calibration', action='store_true',
                        help='Disable baseline calibration for signal extraction')
    parser.add_argument('--lora_type', type=str, default='all',
                        choices=['all', 'app', 'category'],
                        help='Which LoRA types to load: all, app only, or category only')
    parser.add_argument('--debug_signals', action='store_true',
                        help='Print all raw signals for debugging')
    
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


def _strip_image_tags(text: str) -> str:
    """Remove standalone <image> lines from text (for history text-only context)."""
    lines = text.split('\n')
    cleaned = [line for line in lines if line.strip() != '<image>']
    return '\n'.join(cleaned).strip()


def _extract_episode_turns(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ordered (user -> assistant) turns from a multi-turn episode sample."""
    messages = sample.get('messages', [])
    episode_images = sample.get('images', [])
    turns: List[Dict[str, Any]] = []
    image_cursor = 0
    turn_id = 0

    for i, msg in enumerate(messages):
        if msg.get('role') != 'user':
            continue

        query = msg.get('content', '')
        if not query:
            continue

        # Pair this user turn with the nearest following assistant turn.
        assistant_response = None
        for j in range(i + 1, len(messages)):
            role = messages[j].get('role')
            if role == 'assistant':
                assistant_response = messages[j].get('content', '')
                break
            if role == 'user':
                break

        if assistant_response is None:
            continue

        image_tag_count = query.count('<image>')
        if image_tag_count > 0:
            turn_images = episode_images[image_cursor:image_cursor + image_tag_count]
            image_cursor += len(turn_images)
        else:
            turn_images = []

        turn_id += 1
        turns.append({
            'turn_id': turn_id,
            'query': query,
            'images': turn_images,
            'response': assistant_response,
        })

    return turns


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset as raw samples (single-turn or episode-style multi-turn)."""
    data: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

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


def get_template_type(model_type: str) -> str:
    """Get template type based on model type."""
    if 'qwen2-vl' in model_type:
        return 'qwen2-vl'
    elif 'internvl' in model_type:
        return 'internvl2'
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def prepare_model(args) -> tuple:
    """Load and prepare the base model (InternVL2 or Qwen2-VL)."""
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
    
    # Get template based on model type
    template_type = get_template_type(args.model_type)
    logger.info(f"Using template type: {template_type}")
    
    template = get_template(
        template_type,
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
    
    # Set default model path if not specified
    if args.model_path is None:
        args.model_path = MODEL_PATHS.get(args.model_type)
        if args.model_path is None:
            raise ValueError(f"Unknown model type: {args.model_type}. Available: {list(MODEL_PATHS.keys())}")
        logger.info(f"Using default model path for {args.model_type}: {args.model_path}")
    
    # Set default config paths if not specified
    config = CONFIG_PATHS.get(args.model_type, CONFIG_PATHS['internvl2-2b'])
    if args.app_config is None:
        args.app_config = config['app']
        logger.info(f"Using default app config: {args.app_config}")
    if args.category_config is None:
        args.category_config = config['category']
        logger.info(f"Using default category config: {args.category_config}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    episode_output_dir = os.path.join(args.output_dir, 'episode_outputs')
    os.makedirs(episode_output_dir, exist_ok=True)
    time_str = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    model_suffix = 'qwen2vl' if 'qwen2' in args.model_type else 'internvl2'
    output_path = os.path.join(
        episode_output_dir,
        f'retriever_results_{model_suffix}_{args.merge_method}_{args.signal_type}_k{args.top_k}_{time_str}.jsonl'
    )

    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    test_data = load_jsonl(args.test_data)
    logger.info(f"Loaded {len(test_data)} raw episodes")

    # Enforce episode-only input.
    non_episode_indices = [i for i, s in enumerate(test_data) if not isinstance(s.get('messages'), list)]
    if non_episode_indices:
        preview = non_episode_indices[:5]
        raise ValueError(
            f"Episode-only mode is enforced, but found {len(non_episode_indices)} non-episode samples. "
            f"Example indices: {preview}"
        )

    # Limit number of episodes for debugging
    if args.debug or args.num_samples:
        num_samples = args.num_samples or 5
        test_data = test_data[:num_samples]
        logger.info(f"Debug mode: processing {len(test_data)} episodes")
    
    # Load LoRA configs based on lora_type
    config_paths = []
    if args.lora_type in ['all', 'app'] and os.path.exists(args.app_config):
        config_paths.append(args.app_config)
        logger.info(f"Loading app LoRA configs from: {args.app_config}")
    if args.lora_type in ['all', 'category'] and os.path.exists(args.category_config):
        config_paths.append(args.category_config)
        logger.info(f"Loading category LoRA configs from: {args.category_config}")
    
    lora_configs = load_lora_configs(config_paths)
    logger.info(f"Using lora_type={args.lora_type}, loaded {len(lora_configs)} LoRA configs")
    
    if not lora_configs:
        logger.error("No LoRA configs found!")
        return
    
    # Load base model
    model, tokenizer, template = prepare_model(args)
    
    # Initialize LOGO engine
    logger.info("Initializing LOGO engine...")
    # 是否使用基线校准
    use_baseline = not args.no_baseline_calibration
    
    engine = LOGOEngine(
        base_model=model,
        lora_configs=lora_configs,
        top_k=args.top_k,
        signal_type=args.signal_type,
        target_block_idx=args.target_block_idx,
        token_position=args.token_position,
        combination_type=args.combination_type,
        use_baseline_calibration=use_baseline
    )
    
    # Load all LoRAs
    logger.info("Loading all LoRA adapters...")
    engine.load_all_loras()
    
    # Update template model reference
    template.model = engine.model
    
    # Compute baseline signals for calibration (if using norm/entropy signal type)
    if args.signal_type in ['norm', 'entropy'] and use_baseline:
        logger.info("Computing baseline signals for calibration...")
        engine.set_tokenizer(tokenizer)
        engine.compute_baseline(tokenizer=tokenizer)
    else:
        logger.info("Baseline calibration disabled - using raw signals")
    
    # Print engine info
    logger.info(f"LOGO Engine Info: {engine.get_adapter_info()}")
    
    # Project root for resolving paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Run inference
    logger.info("Starting LOGO episode inference...")
    episode_results: List[Dict[str, Any]] = []
    total_turns = 0
    successful_turns = 0

    def infer_one_turn(
        query: str,
        images: List[str],
        history: List[List[str]]
    ) -> str:
        try:
            resolved_images = resolve_image_paths(images, project_root)
            adjusted_query = prepare_query(query, len(resolved_images))
            example = {
                'query': adjusted_query,
                'history': history,
                'system': None,
                'images': resolved_images,
                'audios': [],
                'videos': [],
                'tools': None,
                'objects': None,
            }
            inputs, _ = template.encode(example)

            signal_inputs = {}
            if 'input_ids' in inputs:
                signal_inputs['input_ids'] = torch.tensor([inputs['input_ids']]).to(engine.model.device)
            if 'attention_mask' in inputs:
                signal_inputs['attention_mask'] = torch.tensor([inputs['attention_mask']]).to(engine.model.device)
            if 'inputs_embeds' in inputs:
                signal_inputs['inputs_embeds'] = inputs['inputs_embeds'].unsqueeze(0).to(engine.model.device)
            if 'pixel_values' in inputs:
                signal_inputs['pixel_values'] = inputs['pixel_values'].to(engine.model.device)
            if 'image_grid_thw' in inputs:
                signal_inputs['image_grid_thw'] = inputs['image_grid_thw'].to(engine.model.device)

            selected_loras, weights = engine.process_input(
                **signal_inputs,
                debug_signals=args.debug_signals
            )
            template.model = engine.model

            if args.merge_method == 'mixture':
                lora_mapping = engine.get_lora_mapping(
                    selected_names=selected_loras,
                    weights=weights,
                    batch_size=1,
                    device=engine.model.device
                )
                if resolved_images:
                    response, _ = inference(
                        engine.model,
                        template,
                        adjusted_query,
                        history=history,
                        system=None,
                        images=resolved_images,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        merging_type='mixture',
                        lora_mapping=lora_mapping,
                        mixture_adapter_names=engine.adapter_names
                    )
                else:
                    response, _ = inference(
                        engine.model,
                        template,
                        adjusted_query,
                        history=history,
                        system=None,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        merging_type='mixture',
                        lora_mapping=lora_mapping,
                        mixture_adapter_names=engine.adapter_names
                    )
            else:
                engine.merge_with_weights(
                    adapter_names=selected_loras,
                    weights=weights,
                    combination_type=args.combination_type
                )
                template.model = engine.model
                if resolved_images:
                    response, _ = inference(
                        engine.model,
                        template,
                        adjusted_query,
                        history=history,
                        system=None,
                        images=resolved_images,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature
                    )
                else:
                    response, _ = inference(
                        engine.model,
                        template,
                        adjusted_query,
                        history=history,
                        system=None,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature
                    )
                engine.reset_merged_adapter()
            return response
        except Exception as e:
            logger.error(f"Error processing one turn: {e}")
            import traceback
            traceback.print_exc()
            return f"ERROR: {str(e)}"

    for ep_idx, sample in enumerate(tqdm(test_data, desc="LOGO Episode Inference")):
        episode_id = str(sample.get('episode_id', f'episode_{ep_idx:06d}'))
        turns = _extract_episode_turns(sample)
        rolling_history: List[List[str]] = []
        ground_truths: List[str] = []
        predictions: List[str] = []

        for turn in turns:
            gt = turn.get('response', '')
            pred = infer_one_turn(
                query=turn.get('query', ''),
                images=turn.get('images', []),
                history=rolling_history
            )
            ground_truths.append(gt)
            predictions.append(pred)
            total_turns += 1
            if not pred.startswith('ERROR:'):
                successful_turns += 1

            # Keep multi-turn context with model outputs.
            rolling_history.append([
                _strip_image_tags(turn.get('query', '')),
                pred
            ])

        episode_output = {
            'episode_id': episode_id,
            'mode': 'episode',
            'num_steps': len(ground_truths),
            'ground_truths': ground_truths,
            'predictions': predictions,
        }
        append_to_jsonl(output_path, episode_output)
        episode_results.append(episode_output)

    logger.info(f"\n{'='*60}")
    logger.info("Inference complete!")
    logger.info(f"  Total episodes: {len(episode_results)}")
    logger.info(f"  Total turns: {total_turns}")
    logger.info(f"  Successful turns: {successful_turns}")
    logger.info(f"  Failed turns: {total_turns - successful_turns}")
    logger.info(f"  Results saved to: {output_path}")
    logger.info(f"{'='*60}")

    return episode_results


if __name__ == '__main__':
    args = parse_args()
    run_logo_inference(args)
