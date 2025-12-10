#!/usr/bin/env python3
"""
AlphaQubit 一键诊断脚本 - 分析当前结果找出问题
运行: python diagnose_current_results.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

def print_section(title):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")

def diagnose_results():
    """诊断测试结果"""
    print_section("AlphaQubit 诊断报告")
    
    # 1. 检查结果文件
    summary_file = Path("test_results/paper_comparison_analysis.json")
    if not summary_file.exists():
        print("❌ 未找到 test_results/paper_comparison_analysis.json")
        print("   请先运行评估: python test_finetuned_models.py")
        return False
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    # 2. 总体性能
    print_section("1. 总体性能")
    summary = data['summary']
    print(f"总实验数: {summary['total_experiments']}")
    print(f"平均LER: {summary['average_ler']:.4f} ({summary['average_ler']*100:.2f}%)")
    print(f"论文baseline: 0.0300 (3.00%)")
    print(f"差距: {(summary['average_ler'] - 0.03)*100:.2f} 个百分点")
    print(f"\n最好模型: {data['best_model']['experiment']}")
    print(f"  LER: {data['best_model']['ler']:.4f} ({data['best_model']['ler']*100:.2f}%)")
    print(f"\n最差模型: {data['worst_model']['experiment']}")
    print(f"  LER: {data['worst_model']['ler']:.4f} ({data['worst_model']['ler']*100:.2f}%)")
    
    # 3. 按配置分析
    print_section("2. 按配置分组分析")
    
    by_distance = defaultdict(list)
    by_noise = defaultdict(list)
    
    for exp in data['comparison_by_experiment']:
        name = exp['experiment']
        ler = exp['ler']
        
        # 距离
        for d in ['d3', 'd5', 'd7', 'd9', 'd11', 'd13', 'd15', 'd25']:
            if f'_{d}_' in name:
                by_distance[d].append(ler)
                break
        
        # 噪声
        for r in ['r01', 'r03', 'r05', 'r07', 'r09', 'r15', 'r25', 'r50']:
            if f'_{r}_' in name:
                by_noise[r].append(ler)
                break
    
    print("按码距(Distance):")
    for d in sorted(by_distance.keys()):
        lers = by_distance[d]
        print(f"  {d}: 平均={np.mean(lers):.4f}, 最小={np.min(lers):.4f}, "
              f"最大={np.max(lers):.4f}, 数量={len(lers)}")
    
    print("\n按噪声率(Noise):")
    noise_map = {
        'r01': '1%', 'r03': '3%', 'r05': '5%', 'r07': '7%',
        'r09': '9%', 'r15': '15%', 'r25': '25%', 'r50': '50%'
    }
    for r in ['r01', 'r03', 'r05', 'r07', 'r09', 'r15', 'r25', 'r50']:
        if r in by_noise:
            lers = by_noise[r]
            print(f"  {r} ({noise_map[r]:3s}): 平均={np.mean(lers):.4f}, "
                  f"最小={np.min(lers):.4f}, 最大={np.max(lers):.4f}, 数量={len(lers)}")
    
    # 4. 问题诊断
    print_section("3. 问题诊断")
    
    issues = []
    
    # 检查噪声泛化
    if 'r01' in by_noise and 'r25' in by_noise:
        r01_avg = np.mean(by_noise['r01'])
        r25_avg = np.mean(by_noise['r25'])
        if r25_avg > r01_avg * 3:
            issues.append({
                'severity': 'CRITICAL',
                'problem': '高噪声泛化能力严重不足',
                'evidence': f'r01平均LER={r01_avg:.4f}, r25平均LER={r25_avg:.4f} (差距{r25_avg/r01_avg:.1f}倍)',
                'cause': '预训练数据噪声范围太窄，未覆盖测试噪声范围',
                'solution': '重新生成预训练数据，覆盖0.1%-30%噪声范围'
            })
    
    # 检查码距泛化
    if 'd3' in by_distance and 'd7' in by_distance:
        d3_avg = np.mean(by_distance['d3'])
        d7_avg = np.mean(by_distance['d7'])
        if d7_avg > d3_avg * 2:
            issues.append({
                'severity': 'HIGH',
                'problem': '大码距性能显著下降',
                'evidence': f'd3平均LER={d3_avg:.4f}, d7平均LER={d7_avg:.4f} (差距{d7_avg/d3_avg:.1f}倍)',
                'cause': '模型容量(hidden_dim=256)对大码距不足',
                'solution': '对d>=5使用更大模型 (hidden_dim=384/512)'
            })
    
    # 检查是否接近论文
    best_ler = data['best_model']['ler']
    if best_ler > 0.05:
        issues.append({
            'severity': 'MEDIUM',
            'problem': '最好模型仍未达到论文水平',
            'evidence': f'最好LER={best_ler:.4f}, 论文=0.03',
            'cause': '可能是超参数、训练数据量或模型配置',
            'solution': '检查fine-tuning超参数，增加训练样本'
        })
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"问题 {i} [{issue['severity']}]:")
            print(f"  现象: {issue['problem']}")
            print(f"  证据: {issue['evidence']}")
            print(f"  原因: {issue['cause']}")
            print(f"  解决: {issue['solution']}")
            print()
    else:
        print("✓ 未发现重大问题，模型性能良好")
    
    # 5. 建议
    print_section("4. 改进建议")
    
    if issues:
        print("根据诊断结果，建议按以下优先级执行:")
        print("\n优先级1 (CRITICAL问题):")
        for issue in issues:
            if issue['severity'] == 'CRITICAL':
                print(f"  → {issue['solution']}")
        
        print("\n优先级2 (HIGH问题):")
        for issue in issues:
            if issue['severity'] == 'HIGH':
                print(f"  → {issue['solution']}")
        
        print("\n优先级3 (MEDIUM问题):")
        for issue in issues:
            if issue['severity'] == 'MEDIUM':
                print(f"  → {issue['solution']}")
    
    print("\n通用建议:")
    print("  1. 运行 run_full_pipeline.py 重新训练完整流程")
    print("  2. 监控训练loss曲线确认收敛")
    print("  3. 检查数据平衡性(0/1标签比例)")
    
    print_section("诊断完成")
    print("详细报告已保存到: COMPREHENSIVE_CHECK_REPORT.md")
    print("下一步: 运行 python run_full_pipeline.py 执行完整改进流程")
    
    return True

if __name__ == '__main__':
    success = diagnose_results()
    sys.exit(0 if success else 1)
