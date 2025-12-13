#!/usr/bin/env python3
"""
修复预训练数据噪声范围问题。
生成覆盖测试噪声范围(0.1%-30%)的预训练数据。
"""

import yaml
import subprocess
import sys
from pathlib import Path
import shutil
from datetime import datetime

def backup_config(config_path):
    """备份原始配置"""
    backup_path = config_path.with_suffix(f'.yaml.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    shutil.copy2(config_path, backup_path)
    print(f"✓ 备份原始配置到: {backup_path}")
    return backup_path


def generate_noise_configs():
    """生成不同噪声率的配置"""
    # 基础配置
    base_config = {
        'distance': 3,
        'rounds': 25,
        'cross_talk': 0.01,
        't1': 700,
        'measurement_duration': 100
    }
    
    # 噪声率网格 (覆盖测试范围 0.1% - 30%)
    noise_levels = [
        {'name': 'ultra_low', 'depolarization': 0.001, 'leakage_rate': 0.0005},   # 0.1%
        {'name': 'very_low', 'depolarization': 0.003, 'leakage_rate': 0.0015},    # 0.3%
        {'name': 'low', 'depolarization': 0.005, 'leakage_rate': 0.0025},         # 0.5%
        {'name': 'medium_low', 'depolarization': 0.01, 'leakage_rate': 0.005},    # 1%
        {'name': 'medium', 'depolarization': 0.03, 'leakage_rate': 0.015},        # 3%
        {'name': 'medium_high', 'depolarization': 0.05, 'leakage_rate': 0.025},   # 5%
        {'name': 'high', 'depolarization': 0.07, 'leakage_rate': 0.035},          # 7%
        {'name': 'very_high', 'depolarization': 0.10, 'leakage_rate': 0.050},     # 10%
        {'name': 'extreme', 'depolarization': 0.15, 'leakage_rate': 0.075},       # 15%
        {'name': 'ultra_high', 'depolarization': 0.25, 'leakage_rate': 0.125},    # 25%
        {'name': 'maximum', 'depolarization': 0.30, 'leakage_rate': 0.150},       # 30%
    ]
    
    configs = []
    for noise in noise_levels:
        config = base_config.copy()
        config['depolarization'] = noise['depolarization']
        config['leakage_rate'] = noise['leakage_rate']
        configs.append({
            'name': noise['name'],
            'config': config,
            'noise_percent': noise['depolarization'] * 100
        })
    
    return configs


def update_config_and_generate(config_path, noise_config, samples, output_dir):
    """更新配置并生成数据"""
    
    print(f"\n{'='*80}")
    print(f"生成 {noise_config['name']} 噪声数据 ({noise_config['noise_percent']:.1f}%)")
    print(f"{'='*80}")
    
    # 写入配置
    with open(config_path, 'w') as f:
        yaml.safe_dump(noise_config['config'], f, sort_keys=False)
    
    print(f"配置:")
    print(f"  depolarization: {noise_config['config']['depolarization']}")
    print(f"  leakage_rate: {noise_config['config']['leakage_rate']}")
    
    # 运行生成脚本
    cmd = [
        sys.executable,
        'generate_data.py',
        '--model', 'pauli_plus',
        '--samples', str(samples)
    ]
    
    print(f"\n运行: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # 移动生成的文件到对应的噪声目录
        noise_dir = Path(output_dir) / f"noise_{noise_config['name']}"
        noise_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找最新生成的文件
        output_path = Path('output')
        if output_path.exists():
            files = sorted(output_path.glob('*.npy'), key=lambda p: p.stat().st_mtime, reverse=True)
            if len(files) >= 2:
                # 移动syndromes和logicals文件
                for f in files[:2]:
                    dest = noise_dir / f.name
                    shutil.copy2(f, dest)
                    print(f"  ✓ 复制 {f.name} -> {dest}")
        
        print(f"✓ 完成 {noise_config['name']}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 生成失败: {e}")
        print(f"输出: {e.stdout}")
        print(f"错误: {e.stderr}")
        return False


def main():
    print("="*80)
    print(" 预训练数据噪声范围修复")
    print(" 目标: 生成覆盖 0.1% - 30% 噪声范围的训练数据")
    print("="*80)
    
    # 配置路径
    config_path = Path('configs/pauli_plus.yaml')
    output_dir = Path('pretrain_data_extended')
    
    if not config_path.exists():
        print(f"✗ 配置文件不存在: {config_path}")
        return 1
    
    # 备份原始配置
    backup_path = backup_config(config_path)
    
    try:
        # 生成噪声配置
        noise_configs = generate_noise_configs()
        
        print(f"\n将生成 {len(noise_configs)} 个不同噪声级别的数据集:")
        for cfg in noise_configs:
            print(f"  - {cfg['name']:15s}: {cfg['noise_percent']:5.1f}% 噪声")
        
        # 每个噪声级别生成的样本数
        samples_per_level = 50000  # 可以根据需要调整
        
        print(f"\n每个噪声级别生成 {samples_per_level} 样本，继续执行...")
        
        # 逐个生成
        output_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        
        for cfg in noise_configs:
            if update_config_and_generate(config_path, cfg, samples_per_level, output_dir):
                success_count += 1
        
        print(f"\n{'='*80}")
        print(f"生成完成!")
        print(f"成功: {success_count}/{len(noise_configs)}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*80}")
        
        # 生成使用说明
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write("# Extended Pretraining Data with Varied Noise Levels\n\n")
            f.write("## 噪声级别覆盖\n\n")
            f.write("| 级别 | 去极化率 | 泄漏率 | 等效噪声 |\n")
            f.write("|------|---------|--------|----------|\n")
            for cfg in noise_configs:
                f.write(f"| {cfg['name']:15s} | {cfg['config']['depolarization']:.4f} | "
                       f"{cfg['config']['leakage_rate']:.4f} | {cfg['noise_percent']:5.1f}% |\n")
            f.write(f"\n每个级别包含 {samples_per_level} 个样本\n")
            f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"\n使用说明已保存到: {readme_path}")
        
        # 生成合并脚本
        merge_script = output_dir / "merge_all_data.py"
        with open(merge_script, 'w') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"合并所有噪声级别的数据用于训练\"\"\"
import numpy as np
from pathlib import Path

def merge_all():
    base_dir = Path(__file__).parent
    
    all_syndromes = []
    all_logicals = []
    
    for noise_dir in sorted(base_dir.glob('noise_*')):
        syn_files = list(noise_dir.glob('*syndromes*.npy'))
        log_files = list(noise_dir.glob('*logicals*.npy'))
        
        if syn_files and log_files:
            syn = np.load(syn_files[0])
            log = np.load(log_files[0])
            all_syndromes.append(syn)
            all_logicals.append(log)
            print(f"加载 {noise_dir.name}: {len(syn)} 样本")
    
    if all_syndromes:
        merged_syn = np.concatenate(all_syndromes, axis=0)
        merged_log = np.concatenate(all_logicals, axis=0)
        
        np.save(base_dir / 'merged_syndromes.npy', merged_syn)
        np.save(base_dir / 'merged_logicals.npy', merged_log)
        
        print(f"\\n合并完成!")
        print(f"总样本数: {len(merged_syn)}")
        print(f"输出文件:")
        print(f"  - merged_syndromes.npy")
        print(f"  - merged_logicals.npy")

if __name__ == '__main__':
    merge_all()
""")
        merge_script.chmod(0o755)
        print(f"合并脚本已保存到: {merge_script}")
        print(f"\n运行 'python {merge_script}' 可以合并所有数据集")
        
    finally:
        # 恢复原始配置
        if backup_path.exists():
            shutil.copy2(backup_path, config_path)
            print(f"\n✓ 已恢复原始配置")
    
    print("\n下一步:")
    print("1. 使用扩展的预训练数据重新训练基础模型")
    print("2. 或者直接用于fine-tuning以改善高噪声场景")
    print("3. 运行 comprehensive_diagnosis.py 重新评估")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
