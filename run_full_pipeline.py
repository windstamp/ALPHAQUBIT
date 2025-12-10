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
        """步骤1: 生成扩展噪声范围的预训练数据"""
        self.print_banner("步骤 1/6: 生成预训练数据 (扩展噪声范围)")
        
        if self.args.quick_test:
            self.log("⚡ 快速测试模式: 跳过数据生成")
            return True
        
        # 备份原始配置
        config_path = Path("configs/pauli_plus.yaml")
        if config_path.exists():
            backup_path = config_path.with_suffix('.yaml.backup')
            shutil.copy2(config_path, backup_path)
            self.log(f"已备份配置: {backup_path}")
        
        # 噪声级别配置
        noise_levels = [
            {'depolarization': 0.001, 'name': 'p001'},  # 0.1%
            {'depolarization': 0.005, 'name': 'p005'},  # 0.5%
            {'depolarization': 0.01, 'name': 'p01'},    # 1%
            {'depolarization': 0.03, 'name': 'p03'},    # 3%
            {'depolarization': 0.05, 'name': 'p05'},    # 5%
            {'depolarization': 0.10, 'name': 'p10'},    # 10%
            {'depolarization': 0.15, 'name': 'p15'},    # 15%
            {'depolarization': 0.25, 'name': 'p25'},    # 25%
        ]
        
        samples = 20000 if self.args.quick_test else 50000
        output_dir = Path("pretrain_data_extended")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成每个噪声级别的数据
        for noise in noise_levels:
            self.log(f"生成噪声级别: {noise['name']} ({noise['depolarization']*100}%)")
            
            # 更新配置文件
            import yaml
            config = {
                'distance': 3,
                'rounds': 25,
                'depolarization': noise['depolarization'],
                'leakage_rate': noise['depolarization'] * 0.5,
                'cross_talk': 0.01,
                't1': 700,
                'measurement_duration': 100
            }
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f)
            
            # 运行生成
            success = self.run_command(
                [sys.executable, 'generate_data.py', '--model', 'pauli_plus', '--samples', str(samples)],
                f"生成数据 {noise['name']}",
                check=False
            )
            
            if not success:
                self.log(f"⚠️ {noise['name']} 生成失败,继续下一个", "WARNING")
        
        # 恢复配置
        if backup_path.exists():
            shutil.copy2(backup_path, config_path)
            self.log("已恢复原始配置")
        
        self.log("✓ 预训练数据生成完成")
        return True
    
    def step2_pretrain_model(self):
        """步骤2: 预训练基础模型"""
        self.print_banner("步骤 2/6: 预训练基础模型")
        
        if self.args.skip_pretrain:
            self.log("⏭️ 跳过预训练步骤")
            return True
        
        if self.args.quick_test:
            epochs = 5
            batch_size = 256
        else:
            epochs = 20
            batch_size = 512
        
        # 查找预训练数据
        data_files = list(Path("pretrain_data_extended").rglob("*.npz"))
        if not data_files:
            data_files = list(Path("simulated_data").glob("*.npz"))
        
        if not data_files:
            self.log("⚠️ 未找到预训练数据,跳过预训练", "WARNING")
            return False
        
        # 训练基础模型
        for data_file in data_files[:3] if self.args.quick_test else data_files:
            model_name = data_file.stem.replace('samples_', '')
            
            cmd = [
                sys.executable, 'ai_models/train.py',
                '--npz_file', str(data_file),
                '--epochs', str(epochs),
                '--batch_size', str(batch_size),
                '--lr', '5e-4',
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
        
        # 检查数据目录
        finetune_dir = Path("google_finetune_data/finetune")
        if not finetune_dir.exists():
            self.log(f"✗ Fine-tuning数据目录不存在: {finetune_dir}", "ERROR")
            return False
        
        # 构建命令
        cmd = [
            sys.executable, 'run_finetune_all.py',
            '--data-dir', str(finetune_dir),
            '--output-dir', 'finetuned_models_v2',
            '--batch-size', '256' if self.args.quick_test else '512',
            '--epochs', '10' if self.args.quick_test else '30',
            '--lr', '1e-4',
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
            cmd.extend(['--pretrained', str(pretrained_models[0])])
        
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
        
        test_dir = Path("google_finetune_data/test")
        if not test_dir.exists():
            self.log(f"✗ 测试数据目录不存在: {test_dir}", "ERROR")
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
