#!/usr/bin/env python3
"""
训练进度检查和中间验证脚本
可以在训练过程中运行，检查已完成的模型并进行初步验证

用法:
    python check_training_progress.py --checkpoint-dir checkpoints --data-dir output
    python check_training_progress.py --checkpoint-dir checkpoints --quick  # 快速检查
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

def check_environment():
    """检查运行环境"""
    try:
        import torch
        has_torch = True
        try:
            import torch_npu
            device_type = "NPU"
            device_count = torch_npu.npu.device_count()
        except:
            device_type = "CPU/GPU"
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except:
        has_torch = False
        device_type = "N/A"
        device_count = 0
    
    return has_torch, device_type, device_count

def find_checkpoints(checkpoint_dir):
    """查找所有检查点文件"""
    checkpoints = []
    
    # 查找 .pt 和 .pth 文件
    for ext in ['*.pt', '*.pth']:
        pattern = os.path.join(checkpoint_dir, '**', ext)
        checkpoints.extend(glob.glob(pattern, recursive=True))
    
    # 按修改时间排序
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    return checkpoints

def analyze_checkpoint(checkpoint_path):
    """分析单个检查点"""
    import torch
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'path': checkpoint_path,
            'filename': os.path.basename(checkpoint_path),
            'size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
            'modified': datetime.fromtimestamp(os.path.getmtime(checkpoint_path)).strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # 提取训练信息
        if isinstance(checkpoint, dict):
            info['epoch'] = checkpoint.get('epoch', 'N/A')
            info['train_loss'] = checkpoint.get('train_loss', checkpoint.get('loss', 'N/A'))
            info['train_acc'] = checkpoint.get('train_acc', checkpoint.get('accuracy', 'N/A'))
            info['val_loss'] = checkpoint.get('val_loss', 'N/A')
            info['val_acc'] = checkpoint.get('val_acc', 'N/A')
            info['model_id'] = checkpoint.get('model_id', 'N/A')
            info['has_model_state'] = 'model_state_dict' in checkpoint or 'state_dict' in checkpoint
        else:
            # 可能只保存了 state_dict
            info['has_model_state'] = True
            info['epoch'] = 'N/A (state_dict only)'
        
        return info
    except Exception as e:
        return {
            'path': checkpoint_path,
            'error': str(e)
        }

def load_training_logs(checkpoint_dir):
    """加载训练日志"""
    logs = []
    
    # 查找日志文件
    log_patterns = ['*.log', '*training*.txt', '*progress*.json']
    for pattern in log_patterns:
        for f in glob.glob(os.path.join(checkpoint_dir, '**', pattern), recursive=True):
            logs.append(f)
    
    return logs

def quick_evaluate_model(checkpoint_path, data_dir, num_samples=1000):
    """快速评估单个模型"""
    import torch
    
    # 动态导入模型定义
    try:
        # 尝试从当前目录导入
        sys.path.insert(0, os.path.dirname(checkpoint_path))
        sys.path.insert(0, '.')
        
        # 尝试导入模型
        try:
            from train_full_pipeline import AlphaQubitModel, SurfaceCodeDataset
        except:
            try:
                from train_parallel_models import AlphaQubitModel, SurfaceCodeDataset
            except:
                from train_pipeline_v2 import AlphaQubitModel, SurfaceCodeDataset
    except Exception as e:
        return {'error': f'无法导入模型: {e}'}
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建模型
        model = AlphaQubitModel(input_dim=24, hidden_dim=128, num_layers=4, num_heads=4)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # 加载测试数据
        data_files = glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True)
        if not data_files:
            return {'error': f'未找到数据文件: {data_dir}'}
        
        # 使用第一个文件的一部分数据
        test_file = data_files[0]
        data = np.load(test_file)
        
        if 'data' in data:
            X = data['data'][:num_samples]
            y = data['obs'][:num_samples] if 'obs' in data else data['labels'][:num_samples]
        else:
            X = data['syndromes'][:num_samples]
            y = data['labels'][:num_samples]
        
        # 预处理
        if len(X.shape) == 4:
            # (N, T, 8, 3) -> (N, T, 24)
            N, T, H, W = X.shape
            X = X.reshape(N, T, H * W)
        
        # 填充到25步
        max_rounds = 25
        if X.shape[1] < max_rounds:
            pad_width = ((0, 0), (0, max_rounds - X.shape[1]), (0, 0))
            X = np.pad(X, pad_width, mode='constant', constant_values=0)
        
        # 转换为tensor
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y.astype(int))
        
        # 推理
        with torch.no_grad():
            logits = model(X_tensor)
            preds = (torch.sigmoid(logits) > 0.5).long().squeeze()
        
        # 计算准确率
        accuracy = (preds == y_tensor).float().mean().item()
        
        return {
            'accuracy': accuracy,
            'num_samples': len(y),
            'test_file': os.path.basename(test_file),
        }
        
    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()}

def print_progress_report(checkpoints_info, logs, args):
    """打印进度报告"""
    print("\n" + "=" * 70)
    print("📊 AlphaQubit 训练进度检查报告")
    print("=" * 70)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 环境信息
    has_torch, device_type, device_count = check_environment()
    print(f"\n🖥️  环境: PyTorch={'✓' if has_torch else '✗'}, 设备={device_type}, 数量={device_count}")
    
    # 检查点统计
    print(f"\n📁 检查点目录: {args.checkpoint_dir}")
    print(f"   找到 {len(checkpoints_info)} 个检查点文件")
    
    if not checkpoints_info:
        print("\n⚠️  未找到任何检查点，训练可能尚未开始或目录错误")
        return
    
    # 按模型ID分组
    models = {}
    for info in checkpoints_info:
        if 'error' in info:
            continue
        model_id = info.get('model_id', 'unknown')
        if model_id not in models:
            models[model_id] = []
        models[model_id].append(info)
    
    print(f"\n🔢 检测到 {len(models)} 个不同的模型")
    
    # 显示每个模型的最新状态
    print("\n" + "-" * 70)
    print("📈 各模型训练进度:")
    print("-" * 70)
    print(f"{'模型ID':^10} | {'Epoch':^8} | {'训练Loss':^10} | {'训练Acc':^10} | {'验证Acc':^10}")
    print("-" * 70)
    
    completed_models = 0
    best_acc = 0
    best_model = None
    
    for model_id, infos in sorted(models.items()):
        latest = infos[0]  # 已按时间排序
        
        epoch = latest.get('epoch', 'N/A')
        train_loss = latest.get('train_loss', 'N/A')
        train_acc = latest.get('train_acc', 'N/A')
        val_acc = latest.get('val_acc', 'N/A')
        
        # 格式化显示
        if isinstance(train_loss, (int, float)):
            train_loss = f"{train_loss:.4f}"
        if isinstance(train_acc, (int, float)):
            train_acc = f"{train_acc*100:.1f}%"
            if float(train_acc.rstrip('%')) > best_acc:
                best_acc = float(train_acc.rstrip('%'))
                best_model = model_id
        if isinstance(val_acc, (int, float)):
            val_acc = f"{val_acc*100:.1f}%"
        
        # 检查是否完成 (假设目标是100 epochs)
        if isinstance(epoch, int) and epoch >= 100:
            completed_models += 1
            status = "✅"
        elif isinstance(epoch, int) and epoch >= 50:
            status = "🟡"
        else:
            status = "🔵"
        
        print(f"{status} {str(model_id):^8} | {str(epoch):^8} | {train_loss:^10} | {train_acc:^10} | {val_acc:^10}")
    
    print("-" * 70)
    
    # 总结
    print(f"\n📊 训练总结:")
    print(f"   • 已完成模型: {completed_models}/{len(models)}")
    print(f"   • 最佳准确率: {best_acc:.1f}% (模型 {best_model})")
    
    # 最近的检查点
    print(f"\n🕐 最近更新的检查点:")
    for info in checkpoints_info[:3]:
        if 'error' not in info:
            print(f"   • {info['filename']} - {info['modified']}")
    
    # 快速评估
    if args.evaluate and has_torch and checkpoints_info:
        print(f"\n🧪 快速评估 (使用最新的检查点):")
        latest_checkpoint = checkpoints_info[0]['path']
        print(f"   检查点: {os.path.basename(latest_checkpoint)}")
        
        result = quick_evaluate_model(latest_checkpoint, args.data_dir)
        if 'error' in result:
            print(f"   ❌ 评估失败: {result['error']}")
        else:
            print(f"   ✅ 测试准确率: {result['accuracy']*100:.2f}%")
            print(f"   样本数: {result['num_samples']}")
            print(f"   测试文件: {result['test_file']}")
    
    # 预估完成时间
    if len(models) > 0 and completed_models < len(models):
        # 找到有epoch信息的模型
        epochs_done = []
        for infos in models.values():
            epoch = infos[0].get('epoch')
            if isinstance(epoch, int):
                epochs_done.append(epoch)
        
        if epochs_done:
            avg_epoch = sum(epochs_done) / len(epochs_done)
            remaining_epochs = 100 - avg_epoch
            
            # 估算时间 (假设每个epoch约3分钟)
            est_minutes = remaining_epochs * 3 / 8  # 8个并行
            
            print(f"\n⏱️  预估:")
            print(f"   • 平均进度: {avg_epoch:.0f}/100 epochs ({avg_epoch:.0f}%)")
            print(f"   • 预计剩余时间: ~{est_minutes:.0f} 分钟")
    
    print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(description='检查 AlphaQubit 训练进度')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='检查点目录')
    parser.add_argument('--data-dir', type=str, default='output',
                        help='数据目录 (用于快速评估)')
    parser.add_argument('--quick', action='store_true',
                        help='快速检查模式，只显示统计信息')
    parser.add_argument('--evaluate', action='store_true',
                        help='使用最新检查点进行快速评估')
    parser.add_argument('--watch', action='store_true',
                        help='持续监控模式，每分钟更新一次')
    parser.add_argument('--interval', type=int, default=60,
                        help='监控间隔（秒）')
    
    args = parser.parse_args()
    
    # 检查目录
    if not os.path.exists(args.checkpoint_dir):
        print(f"⚠️  检查点目录不存在: {args.checkpoint_dir}")
        print("   请确保训练已经开始，或检查路径是否正确")
        
        # 尝试常见路径
        common_paths = ['checkpoints', 'output/checkpoints', 'models', 'output/models']
        for p in common_paths:
            if os.path.exists(p):
                print(f"   💡 发现目录: {p}")
        return
    
    def run_check():
        # 查找检查点
        checkpoints = find_checkpoints(args.checkpoint_dir)
        
        if args.quick:
            # 快速模式：只统计
            print(f"📁 找到 {len(checkpoints)} 个检查点")
            if checkpoints:
                latest = checkpoints[0]
                mtime = datetime.fromtimestamp(os.path.getmtime(latest))
                print(f"🕐 最新: {os.path.basename(latest)} ({mtime.strftime('%H:%M:%S')})")
        else:
            # 详细分析
            checkpoints_info = []
            for cp in checkpoints[:20]:  # 只分析最近20个
                info = analyze_checkpoint(cp)
                checkpoints_info.append(info)
            
            # 加载日志
            logs = load_training_logs(args.checkpoint_dir)
            
            # 打印报告
            print_progress_report(checkpoints_info, logs, args)
    
    if args.watch:
        import time
        print(f"👀 持续监控模式 (间隔 {args.interval} 秒，Ctrl+C 退出)")
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                run_check()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n👋 监控结束")
    else:
        run_check()

if __name__ == '__main__':
    main()
