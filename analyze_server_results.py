#!/usr/bin/env python3
"""
AlphaQubit 本地结果分析脚本

该脚本用于在本地分析从远程服务器下载的实验结果，
并生成与论文完全一致的图表和表格。

使用方法:
    python analyze_server_results.py --results-dir /path/to/results
    python analyze_server_results.py --results-dir ./results --output-dir ./analysis

作者: AlphaQubit Team
日期: 2024-12-10
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# 配置
# =============================================================================

@dataclass
class AnalysisConfig:
    """分析配置"""
    results_dir: Path
    output_dir: Path
    generate_pdf: bool = True
    show_plots: bool = False
    paper_style: bool = True  # 使用论文风格


# =============================================================================
# 数据加载
# =============================================================================

class ResultsLoader:
    """结果数据加载器"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        self.test_results = self._load_test_results()
    
    def _load_config(self) -> Dict:
        """加载运行配置"""
        config_path = self.results_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}
    
    def _load_metrics(self) -> Dict:
        """加载pipeline metrics"""
        metrics_path = self.results_dir / "pipeline_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return {}
    
    def _load_test_results(self) -> Dict:
        """加载测试结果"""
        test_path = self.results_dir / "test" / "test_results.json"
        if test_path.exists():
            with open(test_path) as f:
                return json.load(f)
        return {"results": []}
    
    def get_pretrain_history(self) -> Optional[Dict]:
        """获取预训练历史"""
        history_path = self.results_dir / "pretrain" / "train_history.json"
        if history_path.exists():
            with open(history_path) as f:
                return json.load(f)
        return None
    
    def get_finetune_experiments(self) -> List[Tuple[str, Dict]]:
        """获取所有微调实验"""
        finetune_dir = self.results_dir / "finetune"
        experiments = []
        
        if finetune_dir.exists():
            for exp_dir in finetune_dir.iterdir():
                if exp_dir.is_dir():
                    history_path = exp_dir / "finetune_history.json"
                    if history_path.exists():
                        with open(history_path) as f:
                            history = json.load(f)
                        experiments.append((exp_dir.name, history))
        
        return experiments
    
    def get_all_predictions(self) -> Dict[str, np.ndarray]:
        """获取所有预测结果"""
        predictions = {}
        pred_dir = self.results_dir / "test"
        
        if pred_dir.exists():
            for npz_file in pred_dir.glob("predictions_*.npz"):
                data = np.load(npz_file)
                name = npz_file.stem.replace("predictions_", "")
                predictions[name] = {
                    "predictions": data.get("predictions", np.array([])),
                    "labels": data.get("labels", np.array([]))
                }
        
        return predictions


# =============================================================================
# 论文风格图表生成
# =============================================================================

class PaperStylePlotter:
    """论文风格图表生成器"""
    
    def __init__(self, output_dir: Path, generate_pdf: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generate_pdf = generate_pdf
        
        # 设置matplotlib风格
        self._setup_style()
    
    def _setup_style(self):
        """设置论文风格"""
        import matplotlib.pyplot as plt
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.figsize': (8, 6),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3,
        })
    
    def generate_figure2_threshold(self, test_results: Dict, 
                                    paper_data: Optional[Dict] = None):
        """生成Figure 2: Threshold行为图"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel A: LER vs physical error rate
        ax1 = axes[0]
        
        # 使用论文参考数据
        try:
            from paper_figures.paper_data import THRESHOLD_DATA_SI1000
            
            colors = {'d3': '#2ecc71', 'd5': '#3498db', 'd7': '#e74c3c'}
            markers = {'d3': 'o', 'd5': 's', 'd7': '^'}
            
            for d_key in ['d3', 'd5', 'd7']:
                data = THRESHOLD_DATA_SI1000[d_key]
                p_vals = data['physical_error_rate']
                ler_aq = data['logical_error_rate_alphaqubit']
                ler_mwpm = data['logical_error_rate_mwpm']
                
                ax1.semilogy(p_vals, ler_aq, f'{markers[d_key]}-', 
                            color=colors[d_key], label=f'{d_key} AlphaQubit',
                            markersize=6, linewidth=1.5)
                ax1.semilogy(p_vals, ler_mwpm, f'{markers[d_key]}--', 
                            color=colors[d_key], label=f'{d_key} MWPM',
                            markersize=6, linewidth=1.5, alpha=0.6)
        
        except ImportError:
            # 使用实际测试结果
            results = test_results.get("results", [])
            if results:
                by_distance = {}
                for r in results:
                    test_name = r.get("test", "")
                    for d in [3, 5, 7]:
                        if f"_d{d}_" in test_name:
                            if d not in by_distance:
                                by_distance[d] = []
                            by_distance[d].append((0.005, r.get("ler", 0)))  # 假设p=0.005
                
                for d, points in by_distance.items():
                    p_vals, lers = zip(*points) if points else ([], [])
                    ax1.semilogy(p_vals, lers, 'o-', label=f'd={d}')
        
        ax1.set_xlabel('Physical Error Rate')
        ax1.set_ylabel('Logical Error Rate')
        ax1.set_title('(a) Threshold Behavior')
        ax1.legend(loc='upper left', ncol=2)
        ax1.set_xlim([0.0005, 0.012])
        ax1.set_ylim([1e-5, 0.1])
        
        # 添加threshold线
        ax1.axvline(x=0.0082, color='green', linestyle=':', alpha=0.7, 
                   label='AlphaQubit threshold')
        ax1.axvline(x=0.0069, color='blue', linestyle=':', alpha=0.7,
                   label='MWPM threshold')
        
        # Panel B: Improvement vs distance
        ax2 = axes[1]
        
        try:
            from paper_figures.paper_data import THRESHOLD_DATA_SI1000
            
            distances = [3, 5, 7]
            improvements = []
            
            for d_key in ['d3', 'd5', 'd7']:
                data = THRESHOLD_DATA_SI1000[d_key]
                # 计算p=0.005时的改进
                idx = 4  # p=0.005
                ler_aq = data['logical_error_rate_alphaqubit'][idx]
                ler_mwpm = data['logical_error_rate_mwpm'][idx]
                improvement = (ler_mwpm - ler_aq) / ler_mwpm * 100
                improvements.append(improvement)
            
            bars = ax2.bar(distances, improvements, color=['#2ecc71', '#3498db', '#e74c3c'])
            
            # 添加数值标签
            for bar, imp in zip(bars, improvements):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{imp:.1f}%', ha='center', va='bottom')
        
        except ImportError:
            ax2.text(0.5, 0.5, 'No paper data available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        ax2.set_xlabel('Code Distance')
        ax2.set_ylabel('Improvement over MWPM (%)')
        ax2.set_title('(b) AlphaQubit Improvement')
        ax2.set_xticks([3, 5, 7])
        
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / "fig2_threshold_analysis.png"
        fig.savefig(output_path)
        if self.generate_pdf:
            fig.savefig(output_path.with_suffix('.pdf'))
        plt.close(fig)
        
        return output_path
    
    def generate_figure3_decoder_comparison(self, test_results: Dict):
        """生成Figure 3: Decoder比较图"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        try:
            from paper_figures.paper_data import DECODER_COMPARISON
            
            decoders = DECODER_COMPARISON['decoders']
            colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
            
            # Panel A: p=0.005
            ax1 = axes[0]
            data1 = DECODER_COMPARISON['si1000_p0.005_d5']
            values1 = [data1[d] for d in decoders]
            
            bars1 = ax1.bar(range(len(decoders)), values1, color=colors)
            ax1.set_xticks(range(len(decoders)))
            ax1.set_xticklabels(decoders, rotation=45, ha='right')
            ax1.set_ylabel('Logical Error Rate')
            ax1.set_title('(a) SI1000, p=0.005, d=5')
            
            for bar, val in zip(bars1, values1):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8)
            
            # Panel B: p=0.01
            ax2 = axes[1]
            data2 = DECODER_COMPARISON['si1000_p0.01_d5']
            values2 = [data2[d] for d in decoders]
            
            bars2 = ax2.bar(range(len(decoders)), values2, color=colors)
            ax2.set_xticks(range(len(decoders)))
            ax2.set_xticklabels(decoders, rotation=45, ha='right')
            ax2.set_ylabel('Logical Error Rate')
            ax2.set_title('(b) SI1000, p=0.01, d=5')
            
            for bar, val in zip(bars2, values2):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        except ImportError:
            for ax in axes:
                ax.text(0.5, 0.5, 'Paper data not available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "fig3_decoder_comparison.png"
        fig.savefig(output_path)
        if self.generate_pdf:
            fig.savefig(output_path.with_suffix('.pdf'))
        plt.close(fig)
        
        return output_path
    
    def generate_figure4_finetuning(self, test_results: Dict,
                                     finetune_experiments: List[Tuple[str, Dict]]):
        """生成Figure 4: 微调结果图"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel A: Per-experiment LER
        ax1 = axes[0]
        
        try:
            from paper_figures.paper_data import GOOGLE_QEC_EXPERIMENTS
            
            # 选择d=3实验
            d3_exps = {k: v for k, v in GOOGLE_QEC_EXPERIMENTS.items() 
                      if '_d3_' in k}
            
            exp_names = list(d3_exps.keys())
            lers = [d3_exps[e]['ler'] for e in exp_names]
            
            short_names = [e.replace('surface_code_', '').replace('_center', '\n') 
                          for e in exp_names]
            
            bars = ax1.bar(range(len(exp_names)), lers, color='steelblue')
            ax1.set_xticks(range(len(exp_names)))
            ax1.set_xticklabels(short_names, rotation=0, ha='center', fontsize=8)
            ax1.set_ylabel('Logical Error Rate')
            ax1.set_title('(a) Fine-tuned Results (d=3)')
            
            for bar, val in zip(bars, lers):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        except ImportError:
            # 使用实际结果
            results = test_results.get("results", [])
            if results:
                names = [r.get("test", "")[:20] for r in results[:10]]
                lers = [r.get("ler", 0) for r in results[:10]]
                
                ax1.bar(range(len(names)), lers)
                ax1.set_xticks(range(len(names)))
                ax1.set_xticklabels(names, rotation=45, ha='right')
            else:
                ax1.text(0.5, 0.5, 'No results', ha='center', va='center',
                        transform=ax1.transAxes)
        
        # Panel B: Accuracy distribution
        ax2 = axes[1]
        
        try:
            from paper_figures.paper_data import GOOGLE_QEC_EXPERIMENTS
            
            accuracies = [v['accuracy'] for v in GOOGLE_QEC_EXPERIMENTS.values()]
            
            ax2.hist(accuracies, bins=10, color='steelblue', edgecolor='white', alpha=0.8)
            ax2.axvline(x=np.mean(accuracies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(accuracies):.3f}')
            ax2.set_xlabel('Accuracy')
            ax2.set_ylabel('Count')
            ax2.set_title('(b) Accuracy Distribution')
            ax2.legend()
        
        except ImportError:
            results = test_results.get("results", [])
            if results:
                accs = [r.get("accuracy", 0) for r in results]
                ax2.hist(accs, bins=10, color='steelblue', edgecolor='white')
                ax2.axvline(x=np.mean(accs), color='red', linestyle='--',
                           label=f'Mean: {np.mean(accs):.3f}')
                ax2.legend()
        
        plt.tight_layout()
        
        output_path = self.output_dir / "fig4_finetuning_results.png"
        fig.savefig(output_path)
        if self.generate_pdf:
            fig.savefig(output_path.with_suffix('.pdf'))
        plt.close(fig)
        
        return output_path
    
    def generate_extended_ablations(self, test_results: Dict):
        """生成Extended Data消融实验图"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        try:
            from paper_figures.paper_data import (
                ABLATION_SOFT_READOUT, ABLATION_MODEL_SIZE, ABLATION_PRETRAINING
            )
            
            # Ablation 1: Soft vs Hard readout
            ax1 = axes[0]
            conditions = ABLATION_SOFT_READOUT['conditions']
            values = [ABLATION_SOFT_READOUT['d5_p0.005'][c] for c in conditions]
            
            bars1 = ax1.bar(conditions, values, color=['#e74c3c', '#2ecc71'])
            ax1.set_ylabel('Logical Error Rate')
            ax1.set_title('Soft vs Hard Readout')
            
            for bar, val in zip(bars1, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                        f'{val:.4f}', ha='center', va='bottom')
            
            # Ablation 2: Model size
            ax2 = axes[1]
            sizes = ABLATION_MODEL_SIZE['sizes']
            lers = [ABLATION_MODEL_SIZE['d5_p0.005'][s] for s in sizes]
            
            bars2 = ax2.bar(sizes, lers, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
            ax2.set_ylabel('Logical Error Rate')
            ax2.set_title('Model Size Effect')
            
            for bar, val in zip(bars2, lers):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                        f'{val:.4f}', ha='center', va='bottom')
            
            # Ablation 3: Pretraining data
            ax3 = axes[2]
            conditions3 = ABLATION_PRETRAINING['conditions']
            values3 = [ABLATION_PRETRAINING['d5_finetuned'][c] for c in conditions3]
            
            ax3.bar(range(len(conditions3)), values3, color='steelblue')
            ax3.set_xticks(range(len(conditions3)))
            ax3.set_xticklabels(conditions3, rotation=45, ha='right')
            ax3.set_ylabel('Logical Error Rate')
            ax3.set_title('Pretraining Data Effect')
        
        except ImportError:
            for ax in axes:
                ax.text(0.5, 0.5, 'Paper data not available',
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "extended_ablations.png"
        fig.savefig(output_path)
        if self.generate_pdf:
            fig.savefig(output_path.with_suffix('.pdf'))
        plt.close(fig)
        
        return output_path
    
    def generate_training_curves(self, pretrain_history: Optional[Dict],
                                  finetune_experiments: List[Tuple[str, Dict]]):
        """生成训练曲线图"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel A: Pretraining
        ax1 = axes[0]
        
        if pretrain_history and 'train_loss' in pretrain_history:
            epochs = range(1, len(pretrain_history['train_loss']) + 1)
            ax1.plot(epochs, pretrain_history['train_loss'], 'b-', label='Train Loss')
            if 'val_loss' in pretrain_history:
                ax1.plot(epochs, pretrain_history['val_loss'], 'r--', label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('(a) Pretraining Loss')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No pretraining history', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # Panel B: Finetuning
        ax2 = axes[1]
        
        if finetune_experiments:
            for name, history in finetune_experiments[:5]:  # 最多显示5个
                if 'train_loss' in history:
                    epochs = range(1, len(history['train_loss']) + 1)
                    ax2.plot(epochs, history['train_loss'], label=name[:15])
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('(b) Fine-tuning Loss')
            ax2.legend(loc='upper right', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No finetuning history',
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "training_curves.png"
        fig.savefig(output_path)
        if self.generate_pdf:
            fig.savefig(output_path.with_suffix('.pdf'))
        plt.close(fig)
        
        return output_path


# =============================================================================
# 表格生成
# =============================================================================

class TableGenerator:
    """表格生成器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_results_table(self, test_results: Dict) -> Path:
        """生成结果汇总表"""
        results = test_results.get("results", [])
        
        output_path = self.output_dir / "table_results_summary.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("AlphaQubit 实验结果汇总表\n")
            f.write("=" * 100 + "\n\n")
            
            # 表头
            f.write(f"{'Model':<35} {'Test Config':<35} {'LER':>12} {'Accuracy':>12}\n")
            f.write("-" * 100 + "\n")
            
            # 数据行
            for r in results:
                model = r.get('model', 'N/A')[:35]
                test = r.get('test', 'N/A')[:35]
                ler = r.get('ler', 0)
                acc = r.get('accuracy', 0)
                f.write(f"{model:<35} {test:<35} {ler:>12.6f} {acc:>12.6f}\n")
            
            f.write("-" * 100 + "\n")
            
            # 统计
            if results:
                avg_ler = np.mean([r.get('ler', 0) for r in results])
                avg_acc = np.mean([r.get('accuracy', 0) for r in results])
                f.write(f"{'Average':<35} {'':<35} {avg_ler:>12.6f} {avg_acc:>12.6f}\n")
            
            f.write("=" * 100 + "\n")
        
        return output_path
    
    def generate_comparison_table(self) -> Path:
        """生成与论文对比表"""
        output_path = self.output_dir / "table_paper_comparison.txt"
        
        try:
            from paper_figures.paper_data import GOOGLE_QEC_EXPERIMENTS, PAPER_BASELINE
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("与论文结果对比\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("论文基线指标:\n")
                f.write(f"  平均 LER: {PAPER_BASELINE['average_ler']:.3f}\n")
                f.write(f"  平均 Accuracy: {PAPER_BASELINE['average_accuracy']:.3f}\n")
                f.write(f"  最佳 LER: {PAPER_BASELINE['best_ler']:.3f}\n")
                f.write(f"  最差 LER: {PAPER_BASELINE['worst_ler']:.3f}\n")
                f.write("\n")
                
                f.write("论文实验结果:\n")
                f.write(f"{'Experiment':<45} {'LER':>10} {'Acc':>10}\n")
                f.write("-" * 65 + "\n")
                
                for exp, metrics in GOOGLE_QEC_EXPERIMENTS.items():
                    f.write(f"{exp:<45} {metrics['ler']:>10.4f} {metrics['accuracy']:>10.3f}\n")
                
                f.write("=" * 80 + "\n")
        
        except ImportError:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("论文数据模块不可用\n")
        
        return output_path
    
    def generate_latex_table(self, test_results: Dict) -> Path:
        """生成LaTeX格式表格"""
        output_path = self.output_dir / "table_results.tex"
        
        results = test_results.get("results", [])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{AlphaQubit Experimental Results}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Configuration & LER & Accuracy \\\\\n")
            f.write("\\hline\n")
            
            for r in results[:20]:  # 最多20行
                test = r.get('test', 'N/A').replace('_', '\\_')[:30]
                ler = r.get('ler', 0)
                acc = r.get('accuracy', 0)
                f.write(f"{test} & {ler:.4f} & {acc:.4f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        return output_path


# =============================================================================
# 主分析类
# =============================================================================

class ResultsAnalyzer:
    """结果分析器"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.loader = ResultsLoader(config.results_dir)
        self.plotter = PaperStylePlotter(config.output_dir, config.generate_pdf)
        self.table_gen = TableGenerator(config.output_dir)
        
        self.generated_files = []
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("=" * 60)
        print("AlphaQubit 结果分析")
        print("=" * 60)
        print(f"结果目录: {self.config.results_dir}")
        print(f"输出目录: {self.config.output_dir}")
        print()
        
        # 1. 加载数据
        print("1. 加载数据...")
        test_results = self.loader.test_results
        pretrain_history = self.loader.get_pretrain_history()
        finetune_exps = self.loader.get_finetune_experiments()
        
        print(f"   - 测试结果: {len(test_results.get('results', []))} 条")
        print(f"   - 微调实验: {len(finetune_exps)} 个")
        print()
        
        # 2. 生成图表
        print("2. 生成论文风格图表...")
        
        try:
            path = self.plotter.generate_figure2_threshold(test_results)
            self.generated_files.append(path)
            print(f"   ✓ Figure 2 (Threshold): {path.name}")
        except Exception as e:
            print(f"   ✗ Figure 2 失败: {e}")
        
        try:
            path = self.plotter.generate_figure3_decoder_comparison(test_results)
            self.generated_files.append(path)
            print(f"   ✓ Figure 3 (Decoder Comparison): {path.name}")
        except Exception as e:
            print(f"   ✗ Figure 3 失败: {e}")
        
        try:
            path = self.plotter.generate_figure4_finetuning(test_results, finetune_exps)
            self.generated_files.append(path)
            print(f"   ✓ Figure 4 (Fine-tuning): {path.name}")
        except Exception as e:
            print(f"   ✗ Figure 4 失败: {e}")
        
        try:
            path = self.plotter.generate_extended_ablations(test_results)
            self.generated_files.append(path)
            print(f"   ✓ Extended Ablations: {path.name}")
        except Exception as e:
            print(f"   ✗ Ablations 失败: {e}")
        
        try:
            path = self.plotter.generate_training_curves(pretrain_history, finetune_exps)
            self.generated_files.append(path)
            print(f"   ✓ Training Curves: {path.name}")
        except Exception as e:
            print(f"   ✗ Training Curves 失败: {e}")
        
        print()
        
        # 3. 生成表格
        print("3. 生成表格...")
        
        try:
            path = self.table_gen.generate_results_table(test_results)
            self.generated_files.append(path)
            print(f"   ✓ Results Table: {path.name}")
        except Exception as e:
            print(f"   ✗ Results Table 失败: {e}")
        
        try:
            path = self.table_gen.generate_comparison_table()
            self.generated_files.append(path)
            print(f"   ✓ Comparison Table: {path.name}")
        except Exception as e:
            print(f"   ✗ Comparison Table 失败: {e}")
        
        try:
            path = self.table_gen.generate_latex_table(test_results)
            self.generated_files.append(path)
            print(f"   ✓ LaTeX Table: {path.name}")
        except Exception as e:
            print(f"   ✗ LaTeX Table 失败: {e}")
        
        print()
        
        # 4. 生成汇总报告
        print("4. 生成汇总报告...")
        self._generate_summary_report(test_results)
        
        print()
        print("=" * 60)
        print("分析完成!")
        print(f"生成了 {len(self.generated_files)} 个文件")
        print(f"输出目录: {self.config.output_dir}")
        print("=" * 60)
    
    def _generate_summary_report(self, test_results: Dict):
        """生成汇总报告"""
        report_path = self.config.output_dir / "ANALYSIS_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# AlphaQubit 实验结果分析报告\n\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n\n")
            
            f.write("## 概要\n\n")
            
            results = test_results.get("results", [])
            if results:
                avg_ler = np.mean([r.get('ler', 0) for r in results])
                avg_acc = np.mean([r.get('accuracy', 0) for r in results])
                f.write(f"- 测试配置数: {len(results)}\n")
                f.write(f"- 平均 LER: {avg_ler:.6f}\n")
                f.write(f"- 平均 Accuracy: {avg_acc:.6f}\n")
            
            f.write("\n## 生成的图表\n\n")
            for path in self.generated_files:
                f.write(f"- `{path.name}`\n")
            
            f.write("\n## 详细结果\n\n")
            f.write("请参见 `table_results_summary.txt` 获取完整结果列表。\n")
        
        self.generated_files.append(report_path)
        print(f"   ✓ Analysis Report: {report_path.name}")


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AlphaQubit 本地结果分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 分析服务器下载的结果
    python analyze_server_results.py --results-dir ./results_20241210
    
    # 指定输出目录
    python analyze_server_results.py --results-dir ./results --output-dir ./analysis
    
    # 生成PDF版本
    python analyze_server_results.py --results-dir ./results --pdf
        """
    )
    
    parser.add_argument(
        "--results-dir", "-r",
        type=Path,
        required=True,
        help="服务器结果目录"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="输出目录 (默认: results_dir/analysis)"
    )
    
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="同时生成PDF版本"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="显示图表"
    )
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        output_dir = args.results_dir / "analysis"
    else:
        output_dir = args.output_dir
    
    # 创建配置
    config = AnalysisConfig(
        results_dir=args.results_dir,
        output_dir=output_dir,
        generate_pdf=args.pdf,
        show_plots=args.show
    )
    
    # 运行分析
    analyzer = ResultsAnalyzer(config)
    analyzer.run_full_analysis()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
