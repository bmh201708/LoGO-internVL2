#!/usr/bin/env python
"""
LOGO Evaluation Script

对 14 个 App LoRA 测试集和 5 个 Category LoRA 测试集进行完整测评

测评方法：
1. App-level 测试：只加载 14 个 app LoRA 作为候选
2. Category-level 测试：只加载 5 个 category LoRA 作为候选
3. 使用 LOGO 方法进行动态 LoRA 选择和合并
4. 使用 evaluation/test_swift.py 进行评估

使用方法:
    python logo/evaluate_logo.py --gpu 5 --top_k 3 --signal_type norm
    python logo/evaluate_logo.py --gpu 5 --top_k 5 --signal_type entropy
    python logo/evaluate_logo.py --app_only  # 只测试 app-level
    python logo/evaluate_logo.py --category_only  # 只测试 category-level
"""

import os
import sys
import json
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description='LOGO Evaluation Script')
    
    # 基本配置
    parser.add_argument('--gpu', type=int, default=5, help='GPU ID')
    parser.add_argument('--top_k', type=int, default=3, help='Top-K LoRA selection')
    parser.add_argument('--signal_type', type=str, default='norm',
                       choices=['norm', 'entropy', 'uniform'],
                       help='Signal type for LoRA selection')
    parser.add_argument('--merge_method', type=str, default='mixture',
                       choices=['mixture', 'add_weighted_adapter'],
                       help='Merge method: mixture (output-level) or add_weighted_adapter (parameter-level)')
    parser.add_argument('--no_baseline_calibration', action='store_true', default=False,
                       help='Disable baseline calibration (default: False = calibration enabled)')
    
    # 选择测试类型
    parser.add_argument('--app_only', action='store_true', help='Only test app-level')
    parser.add_argument('--category_only', action='store_true', help='Only test category-level')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: output/logo_evaluation_TIMESTAMP)')
    
    # 调试选项
    parser.add_argument('--debug', action='store_true', help='Debug mode (fewer samples)')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without running')
    
    return parser.parse_args()


def get_test_datasets():
    """获取测试数据集列表"""
    app_datasets = [
        'adidas', 'amazon', 'calendar', 'clock', 'decathlon',
        'ebay', 'etsy', 'flipkart', 'gmail', 'google_drive',
        'google_maps', 'kitchen_stories', 'reminder', 'youtube'
    ]
    
    category_datasets = [
        'entertainment', 'lives', 'office', 'shopping', 'traveling'
    ]
    
    return app_datasets, category_datasets


def run_inference(
    test_data: str,
    lora_config: str,
    output_dir: str,
    gpu_id: int,
    top_k: int,
    signal_type: str,
    merge_method: str = 'mixture',
    no_baseline_calibration: bool = False,
    is_category: bool = False,
    debug: bool = False,
    dry_run: bool = False
) -> Optional[str]:
    """运行 LOGO 推理"""
    
    cmd = [
        'python', str(PROJECT_ROOT / 'logo' / 'infer_logo.py'),
        '--test_data', test_data,
        '--output_dir', output_dir,
        '--top_k', str(top_k),
        '--signal_type', signal_type,
        '--merge_method', merge_method,
        '--target_block_idx', '-1',
        '--token_position', 'last',
        '--max_new_tokens', '512',
        '--temperature', '0.0',
    ]
    
    if no_baseline_calibration:
        cmd.append('--no_baseline_calibration')
    
    # 根据类型选择配置
    if is_category:
        cmd.extend(['--app_config', '/dev/null'])
        cmd.extend(['--category_config', str(PROJECT_ROOT / 'config' / 'category_loras_config_internvl2.json')])
    else:
        cmd.extend(['--app_config', str(PROJECT_ROOT / 'config' / 'app_loras_config_internvl2.json')])
        cmd.extend(['--category_config', '/dev/null'])
    
    if debug:
        cmd.extend(['--debug', '--num_samples', '3'])
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # 限制图片最大像素和数量，防止 OOM
    # swift 框架会将小写参数名转为大写读取环境变量 (get_env_args)
    env['MAX_PIXELS'] = '150000'  # 大幅降低以防止 OOM
    env['MAX_NUM'] = '9'  # 限制每个样本最多 6 张图片
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if dry_run:
        print(f"[DRY RUN] CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Running inference: {os.path.basename(test_data)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"[ERROR] Inference failed with return code {result.returncode}")
            return None
        
        # 找到最新生成的结果文件
        output_files = sorted(Path(output_dir).glob('logo_results_*.jsonl'), 
                             key=lambda x: x.stat().st_mtime, reverse=True)
        if output_files:
            return str(output_files[0])
        else:
            print("[ERROR] No output file found")
            return None
            
    except Exception as e:
        print(f"[ERROR] Inference error: {e}")
        return None


def run_evaluation(result_file: str, dry_run: bool = False) -> Tuple[Optional[float], Optional[float]]:
    """运行评估并提取准确率"""
    
    cmd = [
        'python', str(PROJECT_ROOT / 'evaluation' / 'test_swift.py'),
        '--data_path', result_file
    ]
    
    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return None, None
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
        
        # 提取准确率
        step_acc = None
        episode_acc = None
        
        import re
        for line in output.split('\n'):
            if 'Step-level accuracy' in line:
                # 格式: "Step-level accuracy: 45.67%"
                match = re.search(r'(\d+\.?\d*)%?', line.split(':')[-1])
                if match:
                    step_acc = float(match.group(1))
            elif 'Episode-level accuracy' in line:
                match = re.search(r'(\d+\.?\d*)', line.split(':')[-1])
                if match:
                    episode_acc = float(match.group(1))
        
        return step_acc, episode_acc
        
    except Exception as e:
        print(f"[ERROR] Evaluation error: {e}")
        return None, None


def merge_jsonl_files(input_files: List[str], output_file: str) -> bool:
    """合并多个 JSONL 文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for input_file in input_files:
                if os.path.exists(input_file):
                    with open(input_file, 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            line = line.strip()
                            if line:
                                out_f.write(line + '\n')
        return True
    except Exception as e:
        print(f"[ERROR] Failed to merge files: {e}")
        return False


def evaluate_overall(result_files: List[str], output_dir: Path, name: str, dry_run: bool = False) -> Tuple[Optional[float], Optional[float]]:
    """
    合并多个结果文件并计算总体准确率
    
    Args:
        result_files: 结果文件列表
        output_dir: 输出目录
        name: 名称（如 'app_overall' 或 'category_overall'）
        dry_run: 是否只打印命令
    
    Returns:
        (step_accuracy, episode_accuracy)
    """
    if dry_run:
        print(f"[DRY RUN] Would merge {len(result_files)} files and evaluate {name}")
        return None, None
    
    # 过滤存在的文件
    existing_files = [f for f in result_files if os.path.exists(f)]
    
    if not existing_files:
        print(f"[WARNING] No result files found for {name}")
        return None, None
    
    # 合并文件
    merged_file = output_dir / f'{name}_merged.jsonl'
    print(f"\n>>> Merging {len(existing_files)} files for {name} overall evaluation...")
    
    if not merge_jsonl_files(existing_files, str(merged_file)):
        return None, None
    
    # 统计合并后的样本数
    with open(merged_file, 'r') as f:
        sample_count = sum(1 for line in f if line.strip())
    print(f"    Merged file contains {sample_count} samples")
    
    # 运行评估
    step_acc, episode_acc = run_evaluation(str(merged_file), dry_run)
    
    print(f"    {name} Overall - Step: {step_acc}%, Episode: {episode_acc}%")
    
    return step_acc, episode_acc


def main():
    args = parse_args()
    
    # 创建输出目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / 'output' / f'logo_evaluation_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("LOGO Evaluation Script")
    print("=" * 70)
    print(f"GPU ID:       {args.gpu}")
    print(f"Top-K:        {args.top_k}")
    print(f"Signal Type:  {args.signal_type}")
    print(f"Merge Method: {args.merge_method}")
    print(f"No Baseline:  {args.no_baseline_calibration}")
    print(f"Output Dir:   {output_dir}")
    print("=" * 70)
    
    app_datasets, category_datasets = get_test_datasets()
    
    results = {
        'config': {
            'gpu': args.gpu,
            'top_k': args.top_k,
            'signal_type': args.signal_type,
            'timestamp': timestamp,
        },
        'app_results': {},
        'category_results': {},
    }
    
    # =========================================================================
    # Part 1: App-level 评测
    # =========================================================================
    if not args.category_only:
        print("\n" + "=" * 70)
        print("Part 1: App-level Evaluation (14 App LoRAs)")
        print("=" * 70)
        
        app_output_dir = output_dir / 'app_results'
        app_output_dir.mkdir(exist_ok=True)
        
        for dataset in app_datasets:
            test_file = PROJECT_ROOT / 'data' / 'test_data_by_app' / f'{dataset}_train.jsonl'
            
            if not test_file.exists():
                print(f"[WARNING] Test file not found: {test_file}")
                continue
            
            print(f"\n>>> Processing app: {dataset}")
            
            # 运行推理
            result_file = run_inference(
                test_data=str(test_file),
                lora_config=str(PROJECT_ROOT / 'config' / 'app_loras_config_internvl2.json'),
                output_dir=str(app_output_dir),
                gpu_id=args.gpu,
                top_k=args.top_k,
                signal_type=args.signal_type,
                merge_method=args.merge_method,
                no_baseline_calibration=args.no_baseline_calibration,
                is_category=False,
                debug=args.debug,
                dry_run=args.dry_run
            )
            
            if result_file and not args.dry_run:
                # 重命名结果文件
                final_result = app_output_dir / f'{dataset}_results.jsonl'
                os.rename(result_file, final_result)
                
                # 运行评估
                step_acc, episode_acc = run_evaluation(str(final_result), args.dry_run)
                
                results['app_results'][dataset] = {
                    'step_accuracy': step_acc,
                    'episode_accuracy': episode_acc,
                    'result_file': str(final_result)
                }
                
                print(f"    Step Accuracy: {step_acc}%")
                print(f"    Episode Accuracy: {episode_acc}%")
        
        # App-level 总体评测
        if results['app_results'] and not args.dry_run:
            print("\n" + "-" * 70)
            print("App-level Overall Evaluation")
            print("-" * 70)
            
            app_result_files = [
                data['result_file'] for data in results['app_results'].values()
                if 'result_file' in data
            ]
            
            app_overall_step, app_overall_episode = evaluate_overall(
                app_result_files, app_output_dir, 'app_overall', args.dry_run
            )
            
            results['app_overall'] = {
                'step_accuracy': app_overall_step,
                'episode_accuracy': app_overall_episode,
            }
    
    # =========================================================================
    # Part 2: Category-level 评测
    # =========================================================================
    if not args.app_only:
        print("\n" + "=" * 70)
        print("Part 2: Category-level Evaluation (5 Category LoRAs)")
        print("=" * 70)
        
        category_output_dir = output_dir / 'category_results'
        category_output_dir.mkdir(exist_ok=True)
        
        for dataset in category_datasets:
            test_file = PROJECT_ROOT / 'data' / 'test_data_by_category' / f'{dataset}_train.jsonl'
            
            if not test_file.exists():
                print(f"[WARNING] Test file not found: {test_file}")
                continue
            
            print(f"\n>>> Processing category: {dataset}")
            
            # 运行推理
            result_file = run_inference(
                test_data=str(test_file),
                lora_config=str(PROJECT_ROOT / 'config' / 'category_loras_config_internvl2.json'),
                output_dir=str(category_output_dir),
                gpu_id=args.gpu,
                top_k=args.top_k,
                signal_type=args.signal_type,
                merge_method=args.merge_method,
                no_baseline_calibration=args.no_baseline_calibration,
                is_category=True,
                debug=args.debug,
                dry_run=args.dry_run
            )
            
            if result_file and not args.dry_run:
                # 重命名结果文件
                final_result = category_output_dir / f'{dataset}_results.jsonl'
                os.rename(result_file, final_result)
                
                # 运行评估
                step_acc, episode_acc = run_evaluation(str(final_result), args.dry_run)
                
                results['category_results'][dataset] = {
                    'step_accuracy': step_acc,
                    'episode_accuracy': episode_acc,
                    'result_file': str(final_result)
                }
                
                print(f"    Step Accuracy: {step_acc}%")
                print(f"    Episode Accuracy: {episode_acc}%")
        
        # Category-level 总体评测
        if results['category_results'] and not args.dry_run:
            print("\n" + "-" * 70)
            print("Category-level Overall Evaluation")
            print("-" * 70)
            
            category_result_files = [
                data['result_file'] for data in results['category_results'].values()
                if 'result_file' in data
            ]
            
            category_overall_step, category_overall_episode = evaluate_overall(
                category_result_files, category_output_dir, 'category_overall', args.dry_run
            )
            
            results['category_overall'] = {
                'step_accuracy': category_overall_step,
                'episode_accuracy': category_overall_episode,
            }
    
    # =========================================================================
    # 生成汇总报告
    # =========================================================================
    if not args.dry_run:
        # 保存 JSON 结果
        results_file = output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 生成 Markdown 报告
        summary_file = output_dir / 'summary.md'
        with open(summary_file, 'w') as f:
            f.write("# LOGO Evaluation Summary\n\n")
            f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("**Configuration:**\n")
            f.write(f"- GPU ID: {args.gpu}\n")
            f.write(f"- Top-K: {args.top_k}\n")
            f.write(f"- Signal Type: {args.signal_type}\n")
            f.write(f"- Merge Method: {args.merge_method}\n")
            f.write(f"- No Baseline Calibration: {args.no_baseline_calibration}\n\n")
            
            if results['app_results']:
                f.write("---\n\n")
                f.write("## App-level Results (14 App LoRAs)\n\n")
                f.write("| App | Step-level Accuracy | Episode-level Accuracy |\n")
                f.write("|-----|---------------------|------------------------|\n")
                
                total_step = 0
                total_episode = 0
                count = 0
                
                for dataset, data in sorted(results['app_results'].items()):
                    step = data.get('step_accuracy', 'N/A')
                    episode = data.get('episode_accuracy', 'N/A')
                    f.write(f"| {dataset} | {step}% | {episode}% |\n")
                    
                    if step != 'N/A' and episode != 'N/A':
                        total_step += step
                        total_episode += episode
                        count += 1
                
                if count > 0:
                    f.write(f"| **Average** | **{total_step/count:.2f}%** | **{total_episode/count:.2f}%** |\n")
                
                # 添加 App-level 总体准确率
                if 'app_overall' in results:
                    app_overall = results['app_overall']
                    app_step = app_overall.get('step_accuracy', 'N/A')
                    app_episode = app_overall.get('episode_accuracy', 'N/A')
                    f.write(f"| **OVERALL** | **{app_step}%** | **{app_episode}%** |\n")
                f.write("\n")
            
            if results['category_results']:
                f.write("---\n\n")
                f.write("## Category-level Results (5 Category LoRAs)\n\n")
                f.write("| Category | Step-level Accuracy | Episode-level Accuracy |\n")
                f.write("|----------|---------------------|------------------------|\n")
                
                total_step = 0
                total_episode = 0
                count = 0
                
                for dataset, data in sorted(results['category_results'].items()):
                    step = data.get('step_accuracy', 'N/A')
                    episode = data.get('episode_accuracy', 'N/A')
                    f.write(f"| {dataset} | {step}% | {episode}% |\n")
                    
                    if step != 'N/A' and episode != 'N/A':
                        total_step += step
                        total_episode += episode
                        count += 1
                
                if count > 0:
                    f.write(f"| **Average** | **{total_step/count:.2f}%** | **{total_episode/count:.2f}%** |\n")
                
                # 添加 Category-level 总体准确率
                if 'category_overall' in results:
                    cat_overall = results['category_overall']
                    cat_step = cat_overall.get('step_accuracy', 'N/A')
                    cat_episode = cat_overall.get('episode_accuracy', 'N/A')
                    f.write(f"| **OVERALL** | **{cat_step}%** | **{cat_episode}%** |\n")
                f.write("\n")
            
            # 总体汇总表
            f.write("---\n\n")
            f.write("## Overall Summary\n\n")
            f.write("| Level | Step-level Accuracy | Episode-level Accuracy |\n")
            f.write("|-------|---------------------|------------------------|\n")
            
            if 'app_overall' in results:
                app_overall = results['app_overall']
                f.write(f"| App-level OVERALL | {app_overall.get('step_accuracy', 'N/A')}% | {app_overall.get('episode_accuracy', 'N/A')}% |\n")
            
            if 'category_overall' in results:
                cat_overall = results['category_overall']
                f.write(f"| Category-level OVERALL | {cat_overall.get('step_accuracy', 'N/A')}% | {cat_overall.get('episode_accuracy', 'N/A')}% |\n")
            
            f.write("\n")
        
        # 打印最终汇总
        print("\n" + "=" * 70)
        print("Evaluation Complete!")
        print("=" * 70)
        
        if results['app_results']:
            print("\nApp-level Results (14 datasets):")
            for dataset, data in sorted(results['app_results'].items()):
                step = data.get('step_accuracy', 'N/A')
                episode = data.get('episode_accuracy', 'N/A')
                print(f"  {dataset}: Step={step}%, Episode={episode}%")
            
            if 'app_overall' in results:
                app_overall = results['app_overall']
                print(f"\n  >>> App-level OVERALL: Step={app_overall.get('step_accuracy', 'N/A')}%, Episode={app_overall.get('episode_accuracy', 'N/A')}%")
        
        if results['category_results']:
            print("\nCategory-level Results (5 datasets):")
            for dataset, data in sorted(results['category_results'].items()):
                step = data.get('step_accuracy', 'N/A')
                episode = data.get('episode_accuracy', 'N/A')
                print(f"  {dataset}: Step={step}%, Episode={episode}%")
            
            if 'category_overall' in results:
                cat_overall = results['category_overall']
                print(f"\n  >>> Category-level OVERALL: Step={cat_overall.get('step_accuracy', 'N/A')}%, Episode={cat_overall.get('episode_accuracy', 'N/A')}%")
        
        # 总结
        print("\n" + "-" * 70)
        print("SUMMARY (21 metrics total):")
        print("-" * 70)
        print(f"  App-level:      14 datasets + 1 overall = 15 results")
        print(f"  Category-level:  5 datasets + 1 overall =  6 results")
        print(f"  Total: 21 results (each with Step & Episode accuracy)")
        print("-" * 70)
        
        print(f"\nFull report: {summary_file}")
        print(f"Results JSON: {results_file}")
        print("=" * 70)


if __name__ == '__main__':
    main()
