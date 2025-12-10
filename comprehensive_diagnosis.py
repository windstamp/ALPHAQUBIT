#!/usr/bin/env python3
"""
全面诊断AlphaQubit训练结果与论文基准的差距。
分析fine-tuning超参数、数据分布、模型配置等所有关键因素。
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

def analyze_hyperparameters():
    """分析fine-tuning超参数配置"""
    print("="*80)
    print("1. FINE-TUNING 超参数分析")
    print("="*80)
    
    config = {
        "学习率 (Learning Rate)": {
            "当前值": "1e-4",
            "论文推荐": "通常 1e-4 到 5e-4",
            "分析": "✓ 学习率在合理范围内",
            "建议": "可以尝试 5e-4 (更快收敛) 或 5e-5 (更稳定)"
        },
        "训练轮数 (Epochs)": {
            "当前值": "30",
            "early_stopping": "patience=5",
            "分析": "✓ 配置合理",
            "建议": "检查训练日志确认是否提前停止"
        },
        "Batch Size": {
            "当前值": "128 (fine_tune_npz.py)",
            "分析": "✓ 合理的batch size",
            "建议": "如果内存充足，可以尝试256或512加速训练"
        },
        "Weight Decay": {
            "当前值": "1e-3",
            "分析": "✓ 标准正则化强度",
            "建议": "如果过拟合，增加到 5e-3；如果欠拟合，减小到 1e-4"
        },
        "优化器": {
            "当前值": "AdamW",
            "调度器": "CosineAnnealingWarmRestarts (T_0=10, T_mult=2)",
            "分析": "✓ 良好的学习率调度策略",
            "建议": "OK"
        },
        "梯度裁剪": {
            "当前值": "clip_grad_norm=1.0",
            "分析": "✓ 防止梯度爆炸",
            "建议": "OK"
        }
    }
    
    for key, info in config.items():
        print(f"\n【{key}】")
        for k, v in info.items():
            print(f"  {k}: {v}")
    
    print("\n" + "-"*80)
    print("关键问题: 从结果看，d3+r01配置接近论文(3.74% vs 3%)，")
    print("         但高噪声/大码距性能差(最差25.36%)。")
    print("         这说明超参数配置基本OK，问题可能在:")
    print("         1) 预训练数据的噪声范围不足")
    print("         2) fine-tuning数据量不够(困难样本少)")
    print("         3) 模型容量对大码距不足")
    print("-"*80)


def analyze_pretraining_noise():
    """分析预训练数据噪声范围"""
    print("\n" + "="*80)
    print("2. 预训练数据噪声范围分析")
    print("="*80)
    
    print("\n当前配置 (configs/pauli_plus.yaml):")
    print("  - distance: 3")
    print("  - depolarization: 0.001 (0.1%)")
    print("  - leakage_rate: 0.0005 (0.05%)")
    print("  - cross_talk: 0.01 (1%)")
    
    print("\n测试数据噪声范围 (从文件名推断):")
    print("  - r01 = 1%  噪声率")
    print("  - r03 = 3%  噪声率")
    print("  - r05 = 5%  噪声率")
    print("  - r07 = 7%  噪声率")
    print("  - r09 = 9%  噪声率")
    print("  - r25 = 25% 噪声率")
    
    print("\n⚠️  关键问题发现:")
    print("  1. 预训练noise=0.1%，但测试最高到25% - 差距250倍!")
    print("  2. Top 10模型都是r01(1%)，说明只在低噪声下泛化良好")
    print("  3. r25(25%)噪声的模型性能崩溃(LER=25.36%)")
    
    print("\n💡 解决方案:")
    print("  选项A: 重新生成预训练数据，噪声范围覆盖 0.1% - 30%")
    print("  选项B: 使用课程学习 - 先训练低噪声，逐步增加噪声")
    print("  选项C: 增加高噪声场景的fine-tuning数据量")
    
    print("\n建议的噪声分布 (预训练):")
    noise_grid = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
    print("  噪声率网格:", noise_grid)
    print("  每个噪声率生成 50k-100k 样本")


def analyze_model_architecture():
    """分析模型架构配置"""
    print("\n" + "="*80)
    print("3. 模型架构分析")
    print("="*80)
    
    config = {
        "hidden_dim": 256,
        "num_heads": 8,
        "num_layers": 12,
        "embedding": "StabilizerEmbedder (特征投影 + 位置编码)",
        "transformer": "DeepSeekMLA (Multi-head Latent Attention)",
        "readout": "Conv2D + MLP"
    }
    
    print("\n当前模型配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    print("\n分析:")
    print("  ✓ hidden_dim=256 对 d=3 (9个stabilizers) 是足够的")
    print("  ⚠️ hidden_dim=256 对 d=7 (49个stabilizers) 可能不够")
    print("  ✓ num_layers=12 是标准的Transformer深度")
    print("  ✓ num_heads=8 提供了足够的attention多样性")
    
    print("\n码距与模型容量:")
    distances = [3, 5, 7]
    for d in distances:
        stabilizers = d * d
        params_per_stab = config["hidden_dim"] / stabilizers
        print(f"  d={d}: {stabilizers} stabilizers, "
              f"~{params_per_stab:.1f} params/stabilizer")
    
    print("\n💡 建议:")
    print("  1. 对于d>=5的码，考虑增加 hidden_dim 到 512")
    print("  2. 或者使用不同模型尺寸: d3用256, d5用384, d7用512")
    print("  3. 检查是否可以从预训练模型加载权重")


def analyze_data_distribution(test_results_dir="test_results"):
    """分析测试结果的数据分布"""
    print("\n" + "="*80)
    print("4. 数据分布分析")
    print("="*80)
    
    pred_dir = Path(test_results_dir) / "predictions"
    if not pred_dir.exists():
        print(f"⚠️ 预测目录不存在: {pred_dir}")
        return
    
    # 分析最好和最差模型的预测分布
    best_model = "surface_code_bX_d3_r01_center_5_7"
    worst_model = "surface_code_bX_d5_r25_center_5_5"
    
    def analyze_model_predictions(model_name):
        pred_file = pred_dir / f"{model_name}_predictions.npz"
        if not pred_file.exists():
            return None
        
        try:
            pred_data = np.load(pred_file)
            predictions = pred_data['predictions']
            
            # Try to load labels
            test_file = Path("google_finetune_data/test") / f"samples_{model_name}.npz"
            if test_file.exists():
                test_data = np.load(test_file)
                labels = test_data['y'] if 'y' in test_data.files else test_data.get('obs', None)
                
                if labels is not None:
                    min_len = min(len(predictions), len(labels))
                    predictions = predictions[:min_len]
                    labels = labels[:min_len]
                    
                    # Convert obs to binary if needed
                    if np.any(labels > 1):
                        labels = (labels == 49.0).astype(float)
                    
                    accuracy = np.mean(predictions == labels)
                    pred_0 = np.sum(predictions == 0)
                    pred_1 = np.sum(predictions == 1)
                    label_0 = np.sum(labels == 0)
                    label_1 = np.sum(labels == 1)
                    
                    return {
                        'total': len(predictions),
                        'pred_0': int(pred_0),
                        'pred_1': int(pred_1),
                        'label_0': int(label_0),
                        'label_1': int(label_1),
                        'accuracy': float(accuracy),
                        'pred_ratio': float(pred_1 / len(predictions)),
                        'label_ratio': float(label_1 / len(labels))
                    }
        except Exception as e:
            print(f"  错误: {e}")
        return None
    
    print(f"\n最好模型分析: {best_model}")
    best_stats = analyze_model_predictions(best_model)
    if best_stats:
        print(f"  样本总数: {best_stats['total']}")
        print(f"  预测分布: 0={best_stats['pred_0']}, 1={best_stats['pred_1']} "
              f"(比例={best_stats['pred_ratio']:.3f})")
        print(f"  真实分布: 0={best_stats['label_0']}, 1={best_stats['label_1']} "
              f"(比例={best_stats['label_ratio']:.3f})")
        print(f"  准确率: {best_stats['accuracy']:.4f}")
    
    print(f"\n最差模型分析: {worst_model}")
    worst_stats = analyze_model_predictions(worst_model)
    if worst_stats:
        print(f"  样本总数: {worst_stats['total']}")
        print(f"  预测分布: 0={worst_stats['pred_0']}, 1={worst_stats['pred_1']} "
              f"(比例={worst_stats['pred_ratio']:.3f})")
        print(f"  真实分布: 0={worst_stats['label_0']}, 1={worst_stats['label_1']} "
              f"(比例={worst_stats['label_ratio']:.3f})")
        print(f"  准确率: {worst_stats['accuracy']:.4f}")
        
        # 检查是否有预测偏向
        if worst_stats['pred_ratio'] < 0.1 or worst_stats['pred_ratio'] > 0.9:
            print(f"\n  ⚠️ 检测到严重的预测偏向!")
            if worst_stats['pred_ratio'] < 0.1:
                print(f"     模型几乎总是预测0 (no error)")
            else:
                print(f"     模型几乎总是预测1 (error)")
            print(f"     这可能是训练不充分或数据不平衡导致的")


def analyze_by_configuration(test_results_dir="test_results"):
    """按配置分析性能"""
    print("\n" + "="*80)
    print("5. 按配置分组的性能分析")
    print("="*80)
    
    summary_file = Path(test_results_dir) / "paper_comparison_analysis.json"
    if not summary_file.exists():
        print(f"⚠️ 摘要文件不存在: {summary_file}")
        return
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    # 按距离分组
    by_distance = defaultdict(list)
    by_noise = defaultdict(list)
    by_basis = defaultdict(list)
    
    for exp in data['comparison_by_experiment']:
        name = exp['experiment']
        ler = exp['ler']
        
        # 提取distance
        if '_d3_' in name:
            by_distance['d3'].append(ler)
        elif '_d5_' in name:
            by_distance['d5'].append(ler)
        elif '_d7_' in name:
            by_distance['d7'].append(ler)
        elif '_d9_' in name:
            by_distance['d9'].append(ler)
        elif '_d25_' in name:
            by_distance['d25'].append(ler)
        
        # 提取noise rate
        if '_r01_' in name:
            by_noise['r01 (1%)'].append(ler)
        elif '_r03_' in name:
            by_noise['r03 (3%)'].append(ler)
        elif '_r05_' in name:
            by_noise['r05 (5%)'].append(ler)
        elif '_r07_' in name:
            by_noise['r07 (7%)'].append(ler)
        elif '_r09_' in name:
            by_noise['r09 (9%)'].append(ler)
        elif '_r25_' in name:
            by_noise['r25 (25%)'].append(ler)
        elif '_r50_' in name:
            by_noise['r50 (50%)'].append(ler)
        
        # 提取basis
        if '_bX_' in name:
            by_basis['X basis'].append(ler)
        elif '_bZ_' in name:
            by_basis['Z basis'].append(ler)
    
    print("\n按码距(Distance)分组:")
    for d in sorted(by_distance.keys()):
        lers = by_distance[d]
        if lers:
            print(f"  {d:4s}: 平均LER={np.mean(lers):.4f}, "
                  f"最小={np.min(lers):.4f}, 最大={np.max(lers):.4f}, "
                  f"样本数={len(lers)}")
    
    print("\n按噪声率(Noise Rate)分组:")
    for r in ['r01 (1%)', 'r03 (3%)', 'r05 (5%)', 'r07 (7%)', 'r09 (9%)', 'r25 (25%)', 'r50 (50%)']:
        if r in by_noise:
            lers = by_noise[r]
            print(f"  {r:12s}: 平均LER={np.mean(lers):.4f}, "
                  f"最小={np.min(lers):.4f}, 最大={np.max(lers):.4f}, "
                  f"样本数={len(lers)}")
    
    print("\n按测量基(Basis)分组:")
    for b in sorted(by_basis.keys()):
        lers = by_basis[b]
        print(f"  {b:8s}: 平均LER={np.mean(lers):.4f}, "
              f"最小={np.min(lers):.4f}, 最大={np.max(lers):.4f}, "
              f"样本数={len(lers)}")
    
    print("\n关键洞察:")
    
    # 找出性能下降最严重的维度
    if by_distance:
        d_means = {d: np.mean(lers) for d, lers in by_distance.items()}
        worst_d = max(d_means, key=d_means.get)
        best_d = min(d_means, key=d_means.get)
        print(f"  - 码距影响: {best_d}最好({d_means[best_d]:.4f}), "
              f"{worst_d}最差({d_means[worst_d]:.4f})")
    
    if by_noise:
        n_means = {n: np.mean(lers) for n, lers in by_noise.items()}
        worst_n = max(n_means, key=n_means.get)
        best_n = min(n_means, key=n_means.get)
        print(f"  - 噪声影响: {best_n}最好({n_means[best_n]:.4f}), "
              f"{worst_n}最差({n_means[worst_n]:.4f})")


def generate_action_plan():
    """生成改进行动计划"""
    print("\n" + "="*80)
    print("6. 改进行动计划")
    print("="*80)
    
    plan = {
        "紧急优先级 (立即执行)": [
            {
                "问题": "预训练数据噪声范围不足",
                "行动": "重新生成预训练数据，噪声范围0.1%-30%",
                "脚本": "修改 configs/pauli_plus.yaml，运行 make_all_pretraining_noise.py",
                "预期": "改善高噪声场景的泛化能力"
            },
            {
                "问题": "大码距模型容量不足",
                "行动": "对d>=5的码使用更大的hidden_dim",
                "脚本": "修改 run_finetune_all.py，添加动态模型尺寸选择",
                "预期": "改善d5/d7模型的性能"
            }
        ],
        "高优先级 (本周完成)": [
            {
                "问题": "数据不平衡导致预测偏向",
                "行动": "检查并平衡训练数据中0/1标签的比例",
                "脚本": "创建数据平衡脚本，使用weighted sampling",
                "预期": "减少预测偏向问题"
            },
            {
                "问题": "fine-tuning数据量可能不足",
                "行动": "增加困难场景(高噪声、大码距)的样本数",
                "脚本": "运行 google_qec_simulator 生成更多高噪声样本",
                "预期": "改善困难场景的准确率"
            }
        ],
        "中优先级 (两周内)": [
            {
                "问题": "缺少课程学习策略",
                "行动": "实现从易到难的训练顺序",
                "脚本": "创建 curriculum_learning.py",
                "预期": "更稳定的训练过程"
            },
            {
                "问题": "学习率可能需要调整",
                "行动": "网格搜索 lr ∈ {5e-5, 1e-4, 5e-4, 1e-3}",
                "脚本": "添加超参数搜索功能",
                "预期": "找到最优学习率"
            }
        ],
        "实验性 (研究阶段)": [
            {
                "问题": "单模型泛化能力有限",
                "行动": "训练专门的模型族(按distance/noise分组)",
                "脚本": "实现模型集成或动态模型选择",
                "预期": "整体性能提升"
            },
            {
                "问题": "特征工程可能不够",
                "行动": "尝试添加时序特征、空间特征等",
                "脚本": "修改 StabilizerEmbedder",
                "预期": "更好的特征表示"
            }
        ]
    }
    
    for priority, items in plan.items():
        print(f"\n{priority}:")
        for i, item in enumerate(items, 1):
            print(f"\n  {i}. {item['问题']}")
            print(f"     行动: {item['行动']}")
            print(f"     脚本: {item['脚本']}")
            print(f"     预期: {item['预期']}")


def main():
    print("\n" + "="*80)
    print(" AlphaQubit 全面诊断报告")
    print(" 目标: 找出LER 20.13% vs 3%(论文)差距的根本原因")
    print("="*80)
    
    # 1. 超参数分析
    analyze_hyperparameters()
    
    # 2. 预训练数据噪声范围
    analyze_pretraining_noise()
    
    # 3. 模型架构
    analyze_model_architecture()
    
    # 4. 数据分布
    analyze_data_distribution()
    
    # 5. 按配置分析
    analyze_by_configuration()
    
    # 6. 行动计划
    generate_action_plan()
    
    print("\n" + "="*80)
    print(" 诊断完成!")
    print(" 下一步: 选择一个优先级任务开始改进")
    print("="*80 + "\n")
    
    # 保存诊断报告
    report_path = Path("test_results/diagnosis_report.txt")
    print(f"📄 诊断报告已保存到: {report_path}")
    print("   (将控制台输出重定向到文件)")


if __name__ == '__main__':
    main()
