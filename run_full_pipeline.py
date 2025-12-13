#!/usr/bin/env python3
"""
AlphaQubit 完整自动化流程
从噪声数据生成 → 预训练 → Fine-tuning → 评估 → 结果分析

运行: python run_full_pipeline.py [--quick-test]

选项:
  --quick-test    快速测试模式(小数据量)
  --skip-pretrain 跳过预训练,使用现有模型
  --skip-finetune 跳过fine-tuning,使用现有模型
  --only-evaluate 仅运行评估和分析
"""

import argparse
import subprocess
import sys
import os
import time
import json
import shutil
from pathlib import Path
from datetime import datetime

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.log_file = Path("pipeline_log.txt")
        self.results = {
            'start_time': datetime.now().isoformat(),
            'config': vars(args),
            'steps': []
        }
        
        # 备份旧的输出目录，确保从头开始
        if not args.only_evaluate:
            self._backup_old_results()
    
    def _backup_old_results(self):
        """备份旧的输出目录到带时间戳的文件夹"""
        dirs_to_backup = [
            'pretrain_data',
            'pretrained_models', 
            'finetuned_models',
            'finetuned_models_v2',
            'test_results',
            'test_results_v2',
            'output',
            'simulated_data',
        ]
        
        # 检查是否有需要备份的目录
        existing_dirs = [d for d in dirs_to_backup if Path(d).exists()]
        
        if not existing_dirs:
            print("没有旧的结果目录需要备份")
            return
        
        # 创建备份文件夹名称: backup_until_YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = Path(f"backup_until_{timestamp}")
        backup_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"  备份旧结果到: {backup_folder}")
        print(f"{'='*60}")
        
        for dir_name in existing_dirs:
            dir_path = Path(dir_name)
            dest_path = backup_folder / dir_name
            print(f"  移动: {dir_path} -> {dest_path}")
            shutil.move(str(dir_path), str(dest_path))
        
        # 备份旧日志文件
        log_files = list(Path('.').glob('pipeline_*.log')) + list(Path('.').glob('pipeline_*.txt'))
        for log_file in log_files:
            if log_file.exists():
                dest = backup_folder / log_file.name
                print(f"  移动: {log_file} -> {dest}")
                shutil.move(str(log_file), str(dest))
        
        # 备份 pipeline_results.json
        results_json = Path('pipeline_results.json')
        if results_json.exists():
            dest = backup_folder / results_json.name
            print(f"  移动: {results_json} -> {dest}")
            shutil.move(str(results_json), str(dest))
        
        print(f"\n✓ 旧结果已备份到: {backup_folder}")
        print(f"{'='*60}\n")
        
    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def run_command(self, cmd, step_name, check=True):
        """运行命令并记录"""
        self.log(f"开始步骤: {step_name}")
        self.log(f"命令: {' '.join(map(str, cmd))}")
        
        step_start = time.time()
        try:
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            elapsed = time.time() - step_start
            
            self.log(f"✓ {step_name} 完成 (耗时: {elapsed:.1f}秒)", "SUCCESS")
            
            self.results['steps'].append({
                'name': step_name,
                'status': 'success',
                'elapsed': elapsed,
                'command': ' '.join(map(str, cmd))
            })
            
            return True
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - step_start
            self.log(f"✗ {step_name} 失败: {e}", "ERROR")
            self.log(f"输出: {e.stdout}", "ERROR")
            self.log(f"错误: {e.stderr}", "ERROR")
            
            self.results['steps'].append({
                'name': step_name,
                'status': 'failed',
                'elapsed': elapsed,
                'error': str(e),
                'stdout': e.stdout[:1000],
                'stderr': e.stderr[:1000]
            })
            
            if check:
                raise
            return False
    
    def print_banner(self, text):
        """打印横幅"""
        print(f"\n{'='*80}")
        print(f" {text}")
        print(f"{'='*80}\n")
    
    def step1_generate_noise_data(self):
        """步骤1: 生成所有实验的完整噪声数据 (DEM + SI1000 + Soft/IQ)"""
        self.print_banner("步骤 1/6: 生成所有预训练数据 (DEM + SI1000 + Soft/IQ)")
        
        # 使用 make_all_pretraining_noise.py 生成完整数据集
        make_all_script = Path("make_all_pretraining_noise.py")
        
        if not make_all_script.exists():
            self.log("✗ make_all_pretraining_noise.py 不存在", "ERROR")
            return False
        
        if self.args.quick_test:
            # 快速测试模式: 较少样本
            cmd = [
                sys.executable, str(make_all_script),
                '--dem-samples', '10000',
                '--si1000-samples', '10000',
                '--si1000-p-grid', '0.001,0.005,0.01',  # 仅3个p值
                '--soft-shots', '5000',
                '--soft-device', 'auto',
                '--out-dir', 'pretrain_data'
            ]
            self.log("⚡ 快速测试模式: 生成小规模数据")
        else:
            # 完整模式: 论文对齐的数据量
            # Paper: ~8.5M total samples
            # DEM: 200K samples
            # SI1000: 285K samples per (distance, p) × 3 distances × 10 p values = ~8.55M
            # Soft: 100K shots per experiment
            cmd = [
                sys.executable, str(make_all_script),
                '--dem-samples', '200000',
                '--si1000-samples', '285000',
                '--si1000-p-grid', '0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01',
                '--soft-shots', '100000',
                '--soft-device', 'auto',
                '--out-dir', 'pretrain_data'
            ]
            self.log("📊 完整模式: 生成论文对齐的完整数据集 (~8.5M samples)")
            self.log("  - DEM: 200,000 samples")
            self.log("  - SI1000: 285,000 × 3 distances × 10 p-values = ~8.55M samples")
            self.log("  - Soft/IQ: 100,000 shots per experiment")
        
        # 添加实验数据根目录 (如果存在)
        experiment_roots = [
            Path.home() / "work/google_qec3v5_experiment_data",
            Path("experiment_data"),
            Path("google_finetune_data")
        ]
        for root in experiment_roots:
            if root.exists():
                cmd.extend(['--experiment-root', str(root)])
                self.log(f"  发现实验数据目录: {root}")
        
        success = self.run_command(cmd, "生成所有预训练噪声数据")
        
        if success:
            # 检查生成的 MANIFEST
            manifest_path = Path("pretrain_data/MANIFEST.json")
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                item_count = len(manifest.get('items', []))
                self.log(f"✓ 数据生成完成: {item_count} 个数据集")
            else:
                self.log("✓ 预训练数据生成完成")
        
        return success
    
    def step2_pretrain_model(self):
        """步骤2: 预训练基础模型"""
        self.print_banner("步骤 2/6: 预训练基础模型")
        
        if self.args.skip_pretrain:
            self.log("⏭️ 跳过预训练步骤")
            return True
        
        # 论文对齐的超参数
        if self.args.quick_test:
            epochs = 10
            batch_size = 256
            lr = '1e-4'
        else:
            # Paper: 100 epochs, batch_size=256, lr=1e-4, weight_decay=1e-4
            epochs = 100
            batch_size = 256
            lr = '1e-4'
        
        # 查找预训练数据 - 优先使用 pretrain_data 目录
        data_dirs = [
            Path("pretrain_data"),           # 新的统一数据目录
            Path("pretrain_data/si1000"),    # SI1000 数据
            Path("pretrain_data/dem"),       # DEM 数据
            Path("pretrain_data_extended"),  # 旧目录 (兼容)
            Path("simulated_data"),          # 模拟数据
        ]
        
        data_files = []
        for data_dir in data_dirs:
            if data_dir.exists():
                # 查找 .npz 和 .npy 文件
                data_files.extend(list(data_dir.rglob("*.npz")))
                data_files.extend(list(data_dir.rglob("*syndromes*.npy")))
        
        # 去重
        data_files = list(set(data_files))
        data_files.sort()
        
        if not data_files:
            self.log("⚠️ 未找到预训练数据,跳过预训练", "WARNING")
            return False
        
        self.log(f"找到 {len(data_files)} 个数据文件用于预训练")
        
        # 创建输出目录
        Path("pretrained_models").mkdir(parents=True, exist_ok=True)
        
        # 训练基础模型 - 在快速测试模式下限制数量
        files_to_train = data_files[:3] if self.args.quick_test else data_files
        
        for i, data_file in enumerate(files_to_train, 1):
            model_name = data_file.stem.replace('samples_', '').replace('syndromes_', '')
            
            self.log(f"训练 [{i}/{len(files_to_train)}]: {model_name}")
            
            cmd = [
                sys.executable, 'ai_models/train.py',
                '--npz_file', str(data_file),
                '--epochs', str(epochs),
                '--batch_size', str(batch_size),
                '--lr', lr,
                '--weight_decay', '1e-4',  # Paper: weight_decay=1e-4
                '--model-save-path', f'pretrained_models/pretrained_{model_name}.pth'
            ]
            # Add --mla flag if specified
            if self.args.mla:
                cmd.append('--mla')
            
            success = self.run_command(
                cmd,
                f"预训练模型 {model_name}",
                check=False
            )
        
        self.log("✓ 预训练完成")
        return True
    
    def step3_finetune_all(self):
        """步骤3: Fine-tune所有实验"""
        self.print_banner("步骤 3/6: Fine-tuning所有实验")
        
        if self.args.skip_finetune:
            self.log("⏭️ 跳过fine-tuning步骤")
            return True
        
        # 检查数据目录 - 多个可能的位置
        finetune_dirs = [
            Path("google_finetune_data/finetune"),
            Path("pretrain_data"),              # 来自 make_all_pretraining_noise.py 的数据
            Path("simulated_data"),             # 模拟数据
            Path("experiment_data/surface_code"),
        ]
        
        finetune_dir = None
        for d in finetune_dirs:
            if d.exists():
                # 检查是否有 .npz 文件
                npz_files = list(d.rglob("*.npz"))
                if npz_files:
                    finetune_dir = d
                    self.log(f"使用 fine-tuning 数据目录: {d} ({len(npz_files)} 个文件)")
                    break
        
        if finetune_dir is None:
            self.log("✗ 未找到 fine-tuning 数据目录", "ERROR")
            self.log("  请确保以下目录之一存在且包含 .npz 文件:")
            for d in finetune_dirs:
                self.log(f"    - {d}")
            return False
        
        # 论文对齐的超参数
        # Paper: batch_size=128, lr=1e-5, epochs=30, weight_decay=1e-3, patience=5
        if self.args.quick_test:
            batch_size = '128'
            epochs = '10'
            lr = '1e-5'
        else:
            batch_size = '128'  # Paper: 128
            epochs = '30'       # Paper: 30
            lr = '1e-5'         # Paper: 1e-5
        
        # 构建命令
        cmd = [
            sys.executable, 'run_finetune_all.py',
            '--data-dir', str(finetune_dir),
            '--output-dir', 'finetuned_models_v2',
            '--batch-size', batch_size,
            '--epochs', epochs,
            '--lr', lr,
            '--weight-decay', '1e-3',  # Paper: 1e-3
            '--patience', '5',
        ]
        
        if self.args.quick_test:
            cmd.extend(['--limit', '5'])
        
        # Add --mla flag if specified
        if self.args.mla:
            cmd.append('--mla')
        
        # 查找预训练模型
        pretrained_models = list(Path("pretrained_models").glob("*.pth"))
        if pretrained_models:
            # 使用最新的预训练模型
            pretrained_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            cmd.extend(['--pretrained', str(pretrained_models[0])])
            self.log(f"使用预训练模型: {pretrained_models[0]}")
        else:
            self.log("⚠️ 未找到预训练模型,将从头开始训练", "WARNING")
        
        return self.run_command(cmd, "Fine-tuning所有实验")
    
    def step4_evaluate_models(self):
        """步骤4: 评估所有fine-tuned模型"""
        self.print_banner("步骤 4/6: 评估Fine-tuned模型")
        
        # 检查模型目录
        model_dir = Path("finetuned_models_v2")
        if not model_dir.exists():
            model_dir = Path("finetuned_models")
        
        if not model_dir.exists():
            self.log("✗ 未找到fine-tuned模型目录", "ERROR")
            return False
        
        # 检查测试数据目录 - 多个可能的位置
        test_dirs = [
            Path("google_finetune_data/test"),
            Path("pretrain_data"),
            Path("simulated_data"),
            Path("experiment_data/surface_code"),
        ]
        
        test_dir = None
        for d in test_dirs:
            if d.exists():
                npz_files = list(d.rglob("*.npz"))
                if npz_files:
                    test_dir = d
                    self.log(f"使用测试数据目录: {d} ({len(npz_files)} 个文件)")
                    break
        
        if test_dir is None:
            self.log("✗ 未找到测试数据目录", "ERROR")
            self.log("  请确保以下目录之一存在且包含 .npz 文件:")
            for d in test_dirs:
                self.log(f"    - {d}")
            return False
        
        # 运行评估
        cmd = [
            sys.executable, 'test_finetuned_models.py',
            '--model-dir', str(model_dir),
            '--test-dir', str(test_dir),
            '--results-dir', 'test_results_v2',
            '--batch-size', '512',
            '--save-predictions'
        ]
        
        if self.args.quick_test:
            cmd.extend(['--limit', '5'])
        
        # Add --mla flag if specified
        if self.args.mla:
            cmd.append('--mla')
        
        return self.run_command(cmd, "评估模型")
    
    def step5_generate_plots(self):
        """步骤5: 生成对比图表"""
        self.print_banner("步骤 5/6: 生成对比图表和分析")
        
        results_dir = Path("test_results_v2")
        if not results_dir.exists():
            results_dir = Path("test_results")
        
        # 如果test_summary.json不存在,先生成
        summary_file = results_dir / "test_summary.json"
        if not summary_file.exists():
            self.log("生成测试摘要...")
            self.run_command(
                [sys.executable, 'generate_summary_from_predictions.py'],
                "生成测试摘要",
                check=False
            )
        
        # 生成图表和分析
        return self.run_command(
            [sys.executable, 'run_evaluation_and_plot.py'],
            "生成对比图表"
        )
    
    def step6_diagnose(self):
        """步骤6: 诊断结果"""
        self.print_banner("步骤 6/6: 诊断分析")
        
        return self.run_command(
            [sys.executable, 'diagnose_current_results.py'],
            "诊断分析"
        )
    
    def save_results(self):
        """保存流程结果"""
        self.results['end_time'] = datetime.now().isoformat()
        self.results['total_elapsed'] = time.time() - self.start_time
        
        results_file = Path("pipeline_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.log(f"流程结果已保存: {results_file}")
    
    def print_summary(self):
        """打印总结"""
        total_time = time.time() - self.start_time
        
        self.print_banner("流程执行总结")
        
        print(f"总耗时: {total_time/3600:.2f} 小时 ({total_time/60:.1f} 分钟)")
        print(f"\n步骤执行情况:")
        
        for step in self.results['steps']:
            status_icon = "✓" if step['status'] == 'success' else "✗"
            print(f"  {status_icon} {step['name']:40s} {step['elapsed']/60:6.1f}分钟")
        
        success_count = sum(1 for s in self.results['steps'] if s['status'] == 'success')
        total_count = len(self.results['steps'])
        
        print(f"\n成功: {success_count}/{total_count}")
        
        # 查找结果文件
        results_files = [
            "test_results_v2/ler_comparison_with_paper.png",
            "test_results/ler_comparison_with_paper.png",
            "test_results_v2/paper_comparison_analysis.json",
            "test_results/paper_comparison_analysis.json",
        ]
        
        print("\n生成的文件:")
        for f in results_files:
            if Path(f).exists():
                print(f"  ✓ {f}")
        
        print(f"\n详细日志: {self.log_file}")
        print(f"流程结果: pipeline_results.json")
        
        # 显示最终结果
        summary_files = [
            Path("test_results_v2/paper_comparison_analysis.json"),
            Path("test_results/paper_comparison_analysis.json")
        ]
        
        for summary_file in summary_files:
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                
                print(f"\n最终性能:")
                print(f"  平均LER: {data['summary']['average_ler']:.4f} ({data['summary']['average_ler']*100:.2f}%)")
                print(f"  论文baseline: 0.0300 (3.00%)")
                print(f"  最好模型: {data['best_model']['experiment']}")
                print(f"    LER: {data['best_model']['ler']:.4f} ({data['best_model']['ler']*100:.2f}%)")
                break
    
    def run(self):
        """运行完整流程"""
        self.print_banner("AlphaQubit 完整自动化流程")
        self.log(f"配置: {self.args}")
        
        try:
            if not self.args.only_evaluate:
                # 完整流程
                self.step1_generate_noise_data()
                self.step2_pretrain_model()
                self.step3_finetune_all()
            
            # 评估和分析
            self.step4_evaluate_models()
            self.step5_generate_plots()
            self.step6_diagnose()
            
        except KeyboardInterrupt:
            self.log("用户中断执行", "WARNING")
        except Exception as e:
            self.log(f"流程异常: {e}", "ERROR")
            import traceback
            traceback.print_exc()
        finally:
            self.save_results()
            self.print_summary()

def main():
    parser = argparse.ArgumentParser(
        description="AlphaQubit 完整自动化流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程
  python run_full_pipeline.py
  
  # 快速测试(小数据量)
  python run_full_pipeline.py --quick-test
  
  # 仅评估现有模型
  python run_full_pipeline.py --only-evaluate
  
  # 跳过预训练
  python run_full_pipeline.py --skip-pretrain
        """
    )
    
    parser.add_argument('--quick-test', action='store_true',
                        help='快速测试模式(小数据量,少轮数)')
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='跳过预训练步骤')
    parser.add_argument('--skip-finetune', action='store_true',
                        help='跳过fine-tuning步骤')
    parser.add_argument('--only-evaluate', action='store_true',
                        help='仅运行评估和分析(跳过训练)')
    parser.add_argument('--mla', action='store_true',
                        help='使用MLA(Multi-head Latent Attention)模型而非标准transformer')
    
    args = parser.parse_args()
    
    pipeline = Pipeline(args)
    pipeline.run()

if __name__ == '__main__':
    main()
