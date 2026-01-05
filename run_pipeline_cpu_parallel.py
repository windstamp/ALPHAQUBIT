#!/usr/bin/env python3
"""
AlphaQubit CPU 并行训练流程
针对多核 CPU 服务器优化，充分利用所有 CPU 核心

运行: python run_pipeline_cpu_parallel.py

特点:
- 多进程并行训练多个模型
- 自动检测 CPU 核心数
- 优化的 PyTorch CPU 设置
- 进度监控和日志

用法示例:
  python run_pipeline_cpu_parallel.py                    # 完整流程
  python run_pipeline_cpu_parallel.py --quick-test       # 快速测试
  python run_pipeline_cpu_parallel.py --skip-to 3        # 跳到步骤3
  python run_pipeline_cpu_parallel.py --parallel 4       # 4个并行进程
"""

import argparse
import subprocess
import sys
import os
import time
import json
import multiprocessing
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# 设置 CPU 优化环境变量
def setup_cpu_environment():
    """设置 CPU 优化的环境变量"""
    cpu_count = multiprocessing.cpu_count()
    
    # PyTorch CPU 优化
    os.environ['OMP_NUM_THREADS'] = str(max(4, cpu_count // 4))
    os.environ['MKL_NUM_THREADS'] = str(max(4, cpu_count // 4))
    os.environ['NUMEXPR_NUM_THREADS'] = str(max(4, cpu_count // 4))
    
    # 禁用 GPU/NPU 相关
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    print(f"CPU 配置:")
    print(f"  总核心数: {cpu_count}")
    print(f"  OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
    print(f"  MKL_NUM_THREADS: {os.environ['MKL_NUM_THREADS']}")

class CPUParallelPipeline:
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.log_file = Path("pipeline_cpu_parallel.log")
        self.cpu_count = multiprocessing.cpu_count()
        
        # 计算并行进程数
        if args.parallel:
            self.parallel_workers = args.parallel
        else:
            # 自动计算：每个训练进程用 8-16 核
            self.parallel_workers = max(1, self.cpu_count // 12)
        
        self.results = {
            'start_time': datetime.now().isoformat(),
            'config': vars(args),
            'cpu_count': self.cpu_count,
            'parallel_workers': self.parallel_workers,
            'steps': []
        }
        
        self.log(f"初始化 CPU 并行训练")
        self.log(f"  CPU 核心数: {self.cpu_count}")
        self.log(f"  并行进程数: {self.parallel_workers}")
    
    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def print_banner(self, text):
        """打印横幅"""
        print(f"\n{'='*80}")
        print(f" {text}")
        print(f"{'='*80}\n")
    
    def run_command(self, cmd, step_name, check=True):
        """运行单个命令"""
        self.log(f"开始: {step_name}")
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
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - step_start
            self.log(f"✗ {step_name} 失败: {e}", "ERROR")
            self.log(f"输出: {e.stdout[:500] if e.stdout else ''}", "ERROR")
            self.log(f"错误: {e.stderr[:500] if e.stderr else ''}", "ERROR")
            if check:
                raise
            return False, str(e)
    
    def step1_generate_data(self):
        """步骤1: 生成预训练数据"""
        self.print_banner("步骤 1/6: 生成预训练数据")
        
        make_all_script = Path("make_all_pretraining_noise.py")
        if not make_all_script.exists():
            self.log("✗ make_all_pretraining_noise.py 不存在", "ERROR")
            return False
        
        if self.args.quick_test:
            cmd = [
                sys.executable, str(make_all_script),
                '--dem-samples', '10000',
                '--si1000-samples', '10000',
                '--si1000-p-grid', '0.001,0.005,0.01',
                '--soft-shots', '5000',
                '--soft-device', 'auto',
                '--out-dir', 'pretrain_data'
            ]
        else:
            cmd = [
                sys.executable, str(make_all_script),
                '--dem-samples', '200000',
                '--si1000-samples', '285000',
                '--si1000-p-grid', '0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01',
                '--soft-shots', '100000',
                '--soft-device', 'auto',
                '--out-dir', 'pretrain_data'
            ]
        
        success, _ = self.run_command(cmd, "生成预训练数据")
        return success
    
    def step2_pretrain_parallel(self):
        """步骤2: 并行预训练多个模型"""
        self.print_banner("步骤 2/6: 并行预训练模型")
        
        # 收集所有数据文件
        data_dirs = [
            Path("pretrain_data"),
            Path("pretrain_data/si1000"),
            Path("pretrain_data/dem"),
        ]
        
        data_files = []
        for data_dir in data_dirs:
            if data_dir.exists():
                data_files.extend(list(data_dir.rglob("*.npz")))
        
        data_files = list(set(data_files))
        data_files.sort()
        
        if not data_files:
            self.log("⚠️ 未找到预训练数据", "WARNING")
            return False
        
        self.log(f"找到 {len(data_files)} 个数据文件")
        
        # 快速测试模式限制
        if self.args.quick_test:
            data_files = data_files[:5]
        
        # 创建输出目录
        Path("pretrained_models").mkdir(parents=True, exist_ok=True)
        
        # 准备训练任务
        tasks = []
        for data_file in data_files:
            model_name = data_file.stem.replace('samples_', '').replace('syndromes_', '')
            output_path = f'pretrained_models/pretrained_{model_name}.pth'
            
            if Path(output_path).exists() and not self.args.force:
                self.log(f"跳过已存在: {model_name}")
                continue
            
            tasks.append({
                'data_file': str(data_file),
                'model_name': model_name,
                'output_path': output_path,
                'epochs': 10 if self.args.quick_test else 100,
                'batch_size': 256,
                'lr': '1e-4'
            })
        
        if not tasks:
            self.log("所有模型已存在，跳过预训练")
            return True
        
        self.log(f"需要训练 {len(tasks)} 个模型，使用 {self.parallel_workers} 个并行进程")
        
        # 并行训练
        completed = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {executor.submit(train_single_model, task): task for task in tasks}
            
            for future in as_completed(futures):
                task = futures[future]
                try:
                    success = future.result()
                    if success:
                        completed += 1
                        self.log(f"✓ [{completed}/{len(tasks)}] {task['model_name']} 完成")
                    else:
                        failed += 1
                        self.log(f"✗ {task['model_name']} 失败", "ERROR")
                except Exception as e:
                    failed += 1
                    self.log(f"✗ {task['model_name']} 异常: {e}", "ERROR")
        
        self.log(f"预训练完成: {completed} 成功, {failed} 失败")
        return failed == 0
    
    def step3_finetune_parallel(self):
        """步骤3: 并行 Fine-tuning"""
        self.print_banner("步骤 3/6: 并行 Fine-tuning")
        
        # 查找数据
        finetune_dirs = [
            Path("google_finetune_data/finetune"),
            Path("pretrain_data"),
            Path("simulated_data"),
        ]
        
        data_files = []
        for d in finetune_dirs:
            if d.exists():
                data_files.extend(list(d.rglob("*.npz")))
        
        data_files = list(set(data_files))
        data_files.sort()
        
        if not data_files:
            self.log("✗ 未找到 fine-tuning 数据", "ERROR")
            return False
        
        self.log(f"找到 {len(data_files)} 个 fine-tuning 数据文件")
        
        # 快速测试限制
        if self.args.quick_test:
            data_files = data_files[:5]
        
        # 创建输出目录
        Path("finetuned_models").mkdir(parents=True, exist_ok=True)
        
        # 查找预训练模型
        pretrained_models = list(Path("pretrained_models").glob("*.pth"))
        pretrained_path = str(pretrained_models[0]) if pretrained_models else None
        
        if pretrained_path:
            self.log(f"使用预训练模型: {pretrained_path}")
        
        # 准备任务
        tasks = []
        for data_file in data_files:
            model_name = data_file.stem.replace('samples_', '')
            output_path = f'finetuned_models/finetuned_{model_name}.pth'
            
            if Path(output_path).exists() and not self.args.force:
                continue
            
            tasks.append({
                'data_file': str(data_file),
                'model_name': model_name,
                'output_path': output_path,
                'pretrained': pretrained_path,
                'epochs': 10 if self.args.quick_test else 30,
                'batch_size': 128,
                'lr': '1e-5'
            })
        
        if not tasks:
            self.log("所有 fine-tuned 模型已存在，跳过")
            return True
        
        self.log(f"需要 fine-tune {len(tasks)} 个模型")
        
        # 并行 fine-tuning
        completed = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {executor.submit(finetune_single_model, task): task for task in tasks}
            
            for future in as_completed(futures):
                task = futures[future]
                try:
                    success = future.result()
                    if success:
                        completed += 1
                        self.log(f"✓ [{completed}/{len(tasks)}] {task['model_name']} 完成")
                    else:
                        failed += 1
                        self.log(f"✗ {task['model_name']} 失败", "ERROR")
                except Exception as e:
                    failed += 1
                    self.log(f"✗ {task['model_name']} 异常: {e}", "ERROR")
        
        self.log(f"Fine-tuning 完成: {completed} 成功, {failed} 失败")
        return failed == 0
    
    def step4_evaluate(self):
        """步骤4: 评估模型"""
        self.print_banner("步骤 4/6: 评估模型")
        
        cmd = [
            sys.executable, 'run_evaluation_and_plot.py',
            '--model-dir', 'finetuned_models',
            '--output-dir', 'test_results'
        ]
        
        if self.args.quick_test:
            cmd.extend(['--limit', '3'])
        
        success, _ = self.run_command(cmd, "评估模型", check=False)
        return success
    
    def step5_generate_plots(self):
        """步骤5: 生成图表"""
        self.print_banner("步骤 5/6: 生成结果图表")
        
        plot_script = Path("plot_alphaquibit_results.py")
        if not plot_script.exists():
            self.log("⚠️ plot_alphaquibit_results.py 不存在，跳过", "WARNING")
            return True
        
        cmd = [sys.executable, str(plot_script)]
        success, _ = self.run_command(cmd, "生成图表", check=False)
        return success
    
    def step6_analyze(self):
        """步骤6: 分析结果"""
        self.print_banner("步骤 6/6: 分析结果")
        
        analyze_script = Path("analyze_server_results.py")
        if analyze_script.exists():
            cmd = [sys.executable, str(analyze_script)]
            success, _ = self.run_command(cmd, "分析结果", check=False)
            return success
        
        return True
    
    def run(self):
        """运行完整流程"""
        self.print_banner(f"AlphaQubit CPU 并行训练流程")
        self.log(f"配置: {self.args}")
        
        steps = [
            (1, "生成数据", self.step1_generate_data),
            (2, "预训练", self.step2_pretrain_parallel),
            (3, "Fine-tuning", self.step3_finetune_parallel),
            (4, "评估", self.step4_evaluate),
            (5, "生成图表", self.step5_generate_plots),
            (6, "分析结果", self.step6_analyze),
        ]
        
        # 跳到指定步骤
        start_step = self.args.skip_to if self.args.skip_to else 1
        
        for step_num, step_name, step_func in steps:
            if step_num < start_step:
                self.log(f"跳过步骤 {step_num}: {step_name}")
                continue
            
            try:
                success = step_func()
                if not success:
                    self.log(f"步骤 {step_num} 失败，但继续执行", "WARNING")
            except Exception as e:
                self.log(f"步骤 {step_num} 异常: {e}", "ERROR")
        
        # 总结
        elapsed = time.time() - self.start_time
        self.print_banner("流程完成")
        self.log(f"总耗时: {elapsed/3600:.2f} 小时")
        
        # 保存结果
        self.results['end_time'] = datetime.now().isoformat()
        self.results['total_elapsed'] = elapsed
        
        with open('pipeline_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"结果已保存到 pipeline_results.json")


def train_single_model(task):
    """训练单个预训练模型（在子进程中运行）"""
    try:
        cmd = [
            sys.executable, 'ai_models/train.py',
            '--npz_file', task['data_file'],
            '--epochs', str(task['epochs']),
            '--batch_size', str(task['batch_size']),
            '--lr', task['lr'],
            '--weight_decay', '1e-4',
            '--model-save-path', task['output_path'],
        ]
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return True
    except Exception as e:
        print(f"训练失败 {task['model_name']}: {e}")
        return False


def finetune_single_model(task):
    """Fine-tune 单个模型（在子进程中运行）"""
    try:
        cmd = [
            sys.executable, 'ai_models/fine_tune_npz.py',
            '--data', task['data_file'],
            '--output-dir', str(Path(task['output_path']).parent),
            '--epochs', str(task['epochs']),
            '--batch-size', str(task['batch_size']),
            '--lr', task['lr'],
        ]
        
        if task.get('pretrained'):
            cmd.extend(['--pretrained', task['pretrained']])
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return True
    except Exception as e:
        print(f"Fine-tune 失败 {task['model_name']}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='AlphaQubit CPU 并行训练')
    parser.add_argument('--quick-test', action='store_true', help='快速测试模式')
    parser.add_argument('--skip-to', type=int, help='跳到指定步骤 (1-6)')
    parser.add_argument('--parallel', type=int, help='并行进程数 (默认自动计算)')
    parser.add_argument('--force', action='store_true', help='强制重新训练已存在的模型')
    
    args = parser.parse_args()
    
    # 设置 CPU 环境
    setup_cpu_environment()
    
    # 运行流程
    pipeline = CPUParallelPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
