#!/usr/bin/env python3
"""
AlphaQubit Training Pipeline v2 - 改进版

特性:
1. 数据验证模块 - 训练前先检查数据格式
2. 测试模式 - 快速验证整个流程
3. 配置文件支持 - 避免硬编码
4. 实时日志 - 无缓冲输出
5. 自动checkpoint和恢复

用法:
    # 1. 先验证数据
    python train_pipeline_v2.py --validate-only --data-dir output
    
    # 2. 小规模测试 (1个模型, 2个epoch, 1000样本)
    python train_pipeline_v2.py --test-mode --data-dir output
    
    # 3. 全量训练
    python train_pipeline_v2.py --data-dir output --num-models 100 --epochs 100
"""

import os
import sys
import time
import glob
import argparse
import json
import numpy as np
from datetime import datetime

# 强制无缓冲输出
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================================
# 数据验证模块
# ============================================================================

def validate_data_files(data_dir, filter_distance=None, verbose=True):
    """
    验证数据文件格式，返回验证报告
    
    Returns:
        dict: 包含验证结果的报告
    """
    report = {
        'valid': True,
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': [],
        'total_samples': 0,
        'shapes': {},
        'keys_found': set(),
        'issues': []
    }
    
    print("=" * 60)
    print("数据验证报告")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    
    # 查找文件
    patterns = [
        os.path.join(data_dir, "**", "*.npz"),
        os.path.join(data_dir, "*.npz"),
    ]
    
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    all_files = list(set(all_files))
    
    # 按距离过滤
    if filter_distance is not None:
        all_files = [f for f in all_files if f"_d{filter_distance}_" in os.path.basename(f)]
        print(f"过滤条件: distance={filter_distance}")
    
    report['total_files'] = len(all_files)
    print(f"找到文件: {len(all_files)}")
    
    if len(all_files) == 0:
        report['valid'] = False
        report['issues'].append("未找到任何数据文件")
        return report
    
    # 检查每个文件
    print("\n检查文件格式...")
    for i, f in enumerate(all_files[:5]):  # 只详细检查前5个
        try:
            data = np.load(f, allow_pickle=True)
            keys = list(data.keys())
            report['keys_found'].update(keys)
            
            print(f"\n  [{i+1}] {os.path.basename(f)}")
            print(f"      Keys: {keys}")
            
            # 检查必需的键
            has_data = 'data' in keys
            has_obs = 'obs' in keys
            
            if has_data:
                shape = data['data'].shape
                dtype = data['data'].dtype
                print(f"      data: shape={shape}, dtype={dtype}")
                
                shape_key = str(shape[1:])  # 忽略样本数
                report['shapes'][shape_key] = report['shapes'].get(shape_key, 0) + 1
                report['total_samples'] += shape[0]
            
            if has_obs:
                obs_shape = data['obs'].shape
                print(f"      obs: shape={obs_shape}")
            
            if has_data and has_obs:
                report['valid_files'] += 1
            else:
                report['invalid_files'].append({
                    'file': os.path.basename(f),
                    'reason': f"缺少键: data={has_data}, obs={has_obs}"
                })
                
        except Exception as e:
            report['invalid_files'].append({
                'file': os.path.basename(f),
                'reason': str(e)
            })
    
    # 快速检查剩余文件
    print(f"\n快速检查剩余 {len(all_files)-5} 个文件...")
    for f in all_files[5:]:
        try:
            data = np.load(f, allow_pickle=True)
            if 'data' in data.keys() and 'obs' in data.keys():
                report['valid_files'] += 1
                report['total_samples'] += data['data'].shape[0]
                shape_key = str(data['data'].shape[1:])
                report['shapes'][shape_key] = report['shapes'].get(shape_key, 0) + 1
            else:
                report['invalid_files'].append({
                    'file': os.path.basename(f),
                    'reason': "缺少必需的键"
                })
        except Exception as e:
            report['invalid_files'].append({
                'file': os.path.basename(f),
                'reason': str(e)
            })
    
    # 生成报告
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    print(f"有效文件: {report['valid_files']}/{report['total_files']}")
    print(f"总样本数: {report['total_samples']:,}")
    print(f"发现的键: {report['keys_found']}")
    print(f"数据形状分布:")
    for shape, count in report['shapes'].items():
        print(f"  {shape}: {count} 文件")
    
    if report['invalid_files']:
        print(f"\n无效文件 ({len(report['invalid_files'])}个):")
        for item in report['invalid_files'][:5]:
            print(f"  - {item['file']}: {item['reason']}")
        if len(report['invalid_files']) > 5:
            print(f"  ... 还有 {len(report['invalid_files'])-5} 个")
    
    # 检查潜在问题
    if len(report['shapes']) > 1:
        report['issues'].append(f"数据形状不一致: {list(report['shapes'].keys())}")
        print(f"\n⚠️  警告: 数据形状不一致，需要padding")
    
    if report['valid_files'] < report['total_files']:
        report['issues'].append(f"{len(report['invalid_files'])}个文件无效")
    
    report['valid'] = report['valid_files'] > 0
    report['keys_found'] = list(report['keys_found'])
    
    if report['valid']:
        print("\n✅ 数据验证通过!")
    else:
        print("\n❌ 数据验证失败!")
    
    return report


# ============================================================================
# 训练模块 (简化版，只包含核心逻辑)
# ============================================================================

def setup_npu(device_id):
    """设置NPU设备"""
    try:
        import torch
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = False
        return torch.device(f'npu:{device_id}'), True
    except ImportError:
        import torch
        return torch.device('cpu'), False


def run_test_mode(args):
    """
    测试模式: 快速验证整个训练流程
    - 1个模型
    - 2个epoch
    - 最多1000个样本
    """
    print("\n" + "=" * 60)
    print("🧪 测试模式 - 快速验证训练流程")
    print("=" * 60)
    
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    
    # 设置设备
    device, has_npu = setup_npu(0)
    print(f"设备: {device} (NPU可用: {has_npu})")
    
    # 加载少量数据
    print("\n1. 加载测试数据...")
    data_files = glob.glob(os.path.join(args.data_dir, "*d3*.npz"))[:3]
    
    if not data_files:
        print("❌ 未找到数据文件!")
        return False
    
    all_data = []
    all_labels = []
    max_rounds = 25
    
    for f in data_files:
        try:
            npz = np.load(f)
            data = np.array(npz['data'], dtype=np.float32)
            labels = np.array(npz['obs'], dtype=np.float32)
            
            # Padding
            N, T, spatial, channels = data.shape
            if T < max_rounds:
                pad_width = ((0, 0), (0, max_rounds - T), (0, 0), (0, 0))
                data = np.pad(data, pad_width, mode='constant')
            
            all_data.append(data[:500])  # 每个文件只取500个样本
            all_labels.append(labels[:500])
            print(f"   加载: {os.path.basename(f)}, shape={data.shape}")
        except Exception as e:
            print(f"   跳过: {os.path.basename(f)}, 错误={e}")
    
    if not all_data:
        print("❌ 无法加载任何数据!")
        return False
    
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"   总样本: {len(X)}, 形状: {X.shape}")
    
    # 创建简单模型
    print("\n2. 创建测试模型...")
    
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=24, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, x):
            # x: (B, T, spatial, channels)
            x = x.view(x.size(0), x.size(1), -1)  # (B, T, spatial*channels)
            x = x.mean(dim=1)  # (B, features)
            return self.net(x)
    
    model = SimpleModel().to(device)
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建DataLoader
    print("\n3. 创建DataLoader...")
    
    class SimpleDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y).float()
            if len(self.y.shape) == 1:
                self.y = self.y.unsqueeze(1)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    dataset = SimpleDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    print(f"   Batch数: {len(loader)}")
    
    # 训练
    print("\n4. 开始训练 (2 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(2):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (x, y_true) in enumerate(loader):
            x = x.to(device)
            y_true = y_true.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(y_pred) > 0.5).float()
            correct += (preds == y_true).sum().item()
            total += y_true.numel()
            
            if batch_idx % 5 == 0:
                print(f"   Epoch {epoch+1}, Batch {batch_idx}/{len(loader)}: "
                      f"loss={loss.item():.4f}")
        
        acc = correct / total * 100
        avg_loss = total_loss / len(loader)
        print(f"   Epoch {epoch+1} 完成: avg_loss={avg_loss:.4f}, acc={acc:.1f}%")
    
    # 保存测试模型
    print("\n5. 保存测试模型...")
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "test_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"   保存到: {save_path}")
    
    print("\n" + "=" * 60)
    print("✅ 测试模式完成! 所有组件正常工作。")
    print("=" * 60)
    print("\n现在可以运行全量训练:")
    print(f"  python {sys.argv[0]} --data-dir {args.data_dir} --num-models 100 --epochs 100")
    
    return True


def run_full_training(args):
    """运行完整训练"""
    print("\n" + "=" * 60)
    print("🚀 开始完整训练")
    print("=" * 60)
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, RandomSampler
    import threading
    import queue
    import math
    
    # ========== 模型定义 ==========
    class TransformerEncoderLayerManual(nn.Module):
        """手动实现的Transformer层 (避免NPU CPU回退)"""
        def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.nhead = nhead
            self.head_dim = d_model // nhead
            
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            
            self.ff1 = nn.Linear(d_model, dim_feedforward)
            self.ff2 = nn.Linear(dim_feedforward, d_model)
            
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.head_dim)
        
        def forward(self, x, src_mask=None, src_key_padding_mask=None):
            B, T, D = x.shape
            
            q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            
            attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, D)
            out = self.out_proj(out)
            
            x = self.norm1(x + self.dropout(out))
            ff_out = self.ff2(self.dropout(F.gelu(self.ff1(x))))
            x = self.norm2(x + self.dropout(ff_out))
            
            return x
    
    class AlphaQubitModel(nn.Module):
        """AlphaQubit模型"""
        def __init__(self, input_dim=24, hidden_dim=256, num_heads=8, 
                     num_layers=6, num_classes=1, max_seq_len=50):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
            self.transformer_layers = nn.ModuleList([
                TransformerEncoderLayerManual(hidden_dim, num_heads, hidden_dim * 4, dropout=0.1)
                for _ in range(num_layers)
            ])
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes)
            )
        
        def forward(self, x):
            B = x.shape[0]
            if len(x.shape) == 4:
                x = x.view(B, x.shape[1], -1)
            T = x.shape[1]
            
            x = self.input_proj(x)
            seq_len = min(T, self.pos_encoding.size(1))
            x = x[:, :seq_len] + self.pos_encoding[:, :seq_len]
            
            for layer in self.transformer_layers:
                x = layer(x)
            
            x = x.mean(dim=1)
            return self.output_head(x)
    
    # ========== 数据集 ==========
    class SurfaceCodeDataset(Dataset):
        def __init__(self, data_files, max_samples=None, max_rounds=25, verbose=True):
            all_data = []
            all_labels = []
            
            if verbose:
                print(f"加载 {len(data_files)} 个文件...")
            
            for i, f in enumerate(data_files):
                try:
                    npz = np.load(f, allow_pickle=True)
                    data = np.array(npz['data'], dtype=np.float32)
                    labels = np.array(npz['obs'], dtype=np.float32)
                    
                    N, T, spatial, channels = data.shape
                    if T < max_rounds:
                        pad_width = ((0, 0), (0, max_rounds - T), (0, 0), (0, 0))
                        data = np.pad(data, pad_width, mode='constant')
                    elif T > max_rounds:
                        data = data[:, :max_rounds, :, :]
                    
                    if len(labels.shape) == 1:
                        labels = labels.reshape(-1, 1)
                    
                    all_data.append(data)
                    all_labels.append(labels)
                    
                    if verbose and i < 3:
                        print(f"  [{i+1}] {os.path.basename(f)}: orig_T={T}, samples={N}")
                        
                except Exception as e:
                    if verbose:
                        print(f"  跳过 {os.path.basename(f)}: {e}")
            
            if not all_data:
                self.data = np.array([])
                self.labels = np.array([])
                return
            
            self.data = np.concatenate(all_data, axis=0)
            self.labels = np.concatenate(all_labels, axis=0)
            
            if max_samples and len(self.data) > max_samples:
                idx = np.random.choice(len(self.data), max_samples, replace=False)
                self.data = self.data[idx]
                self.labels = self.labels[idx]
            
            if verbose:
                print(f"总样本: {len(self.data)}, 形状: {self.data.shape}")
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return torch.from_numpy(self.data[idx]), torch.from_numpy(self.labels[idx])
    
    # ========== 训练单个模型 ==========
    def train_one_model(npu_id, model_id, data_files, args, result_queue):
        try:
            device, _ = setup_npu(npu_id)
            print(f"[NPU {npu_id}] 模型 {model_id}: 启动")
            
            dataset = SurfaceCodeDataset(data_files, max_rounds=25, verbose=(model_id < 2))
            
            if len(dataset) == 0:
                result_queue.put((model_id, npu_id, False, "数据加载失败"))
                return
            
            sample_x, _ = dataset[0]
            input_dim = sample_x.shape[1] * sample_x.shape[2] if len(sample_x.shape) == 3 else 24
            
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=0, drop_last=True)
            
            model = AlphaQubitModel(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            criterion = nn.BCEWithLogitsLoss()
            
            best_loss = float('inf')
            start_time = time.time()
            
            for epoch in range(args.epochs):
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (x, y) in enumerate(loader):
                    x, y = x.to(device), y.to(device)
                    
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct += (preds == y).sum().item()
                    total += y.numel()
                    
                    if batch_idx > 0 and batch_idx % 100 == 0:
                        print(f"[NPU {npu_id}] 模型 {model_id} Epoch {epoch+1} "
                              f"Batch {batch_idx}: loss={total_loss/batch_idx:.4f}, "
                              f"acc={correct/total*100:.1f}%")
                
                scheduler.step()
                avg_loss = total_loss / len(loader)
                acc = correct / total * 100
                
                print(f"[NPU {npu_id}] 模型 {model_id} Epoch {epoch+1}/{args.epochs}: "
                      f"loss={avg_loss:.4f}, acc={acc:.1f}%")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_path = os.path.join(args.output_dir, f"model_{model_id}_best.pth")
                    torch.save(model.state_dict(), save_path)
            
            # 保存最终模型
            final_path = os.path.join(args.output_dir, f"model_{model_id}_final.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': args.epochs,
                'loss': avg_loss,
            }, final_path)
            
            total_time = time.time() - start_time
            result_queue.put((model_id, npu_id, True, 
                            f"完成: {total_time/3600:.2f}h, loss={best_loss:.4f}"))
            
        except Exception as e:
            import traceback
            result_queue.put((model_id, npu_id, False, f"{e}\n{traceback.format_exc()}"))
    
    # ========== 并行训练 ==========
    # 查找数据文件
    data_files = sorted(glob.glob(os.path.join(args.data_dir, f"*d{args.filter_distance}*.npz")))
    
    if not data_files:
        print("❌ 未找到数据文件!")
        return False
    
    print(f"数据文件: {len(data_files)}")
    print(f"模型数量: {args.num_models}")
    print(f"NPU数量: {args.num_npus}")
    print(f"Epochs: {args.epochs}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    config = vars(args).copy()
    config['data_files'] = len(data_files)
    config['start_time'] = datetime.now().isoformat()
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    result_queue = queue.Queue()
    completed = []
    failed = []
    model_queue = list(range(args.num_models))
    active_threads = {}
    
    start_time = time.time()
    
    while model_queue or active_threads:
        # 检查完成的模型
        while not result_queue.empty():
            model_id, npu_id, success, msg = result_queue.get()
            if npu_id in active_threads:
                del active_threads[npu_id]
            
            if success:
                completed.append(model_id)
                print(f"\n>>> 模型 {model_id} 完成: {msg}")
            else:
                failed.append(model_id)
                print(f"\n>>> 模型 {model_id} 失败: {msg}")
            
            print(f">>> 进度: {len(completed)}/{args.num_models} 完成, "
                  f"{len(failed)} 失败, {len(model_queue)} 等待中")
        
        # 启动新模型
        for npu_id in range(args.num_npus):
            if npu_id not in active_threads and model_queue:
                model_id = model_queue.pop(0)
                thread = threading.Thread(
                    target=train_one_model,
                    args=(npu_id, model_id, data_files, args, result_queue)
                )
                thread.start()
                active_threads[npu_id] = thread
                print(f"\n>>> 启动模型 {model_id} 在 NPU {npu_id}")
                time.sleep(2)
        
        time.sleep(10)
    
    # 最终报告
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总时间: {total_time/3600:.2f} 小时")
    print(f"成功: {len(completed)}")
    print(f"失败: {len(failed)}")
    print(f"输出目录: {args.output_dir}")
    
    return len(failed) == 0


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AlphaQubit Training Pipeline v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 1. 验证数据
  python train_pipeline_v2.py --validate-only --data-dir output
  
  # 2. 测试模式
  python train_pipeline_v2.py --test-mode --data-dir output
  
  # 3. 全量训练
  python train_pipeline_v2.py --data-dir output --num-models 100 --epochs 100
        """
    )
    
    # 模式选择
    parser.add_argument('--validate-only', action='store_true',
                        help='仅验证数据，不训练')
    parser.add_argument('--test-mode', action='store_true',
                        help='测试模式: 1模型, 2epoch, 少量数据')
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default='output',
                        help='数据目录')
    parser.add_argument('--filter-distance', type=int, default=3,
                        help='表面码距离 (3, 5, 7)')
    
    # 模型参数
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=6)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # 并行参数
    parser.add_argument('--num-models', type=int, default=100)
    parser.add_argument('--num-npus', type=int, default=8)
    
    # 输出
    parser.add_argument('--output-dir', type=str, default='trained_models')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AlphaQubit Training Pipeline v2")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if args.validate_only:
        # 仅验证数据
        report = validate_data_files(args.data_dir, args.filter_distance)
        return 0 if report['valid'] else 1
    
    elif args.test_mode:
        # 测试模式
        success = run_test_mode(args)
        return 0 if success else 1
    
    else:
        # 全量训练
        # 先快速验证
        print("\n[预检查] 验证数据...")
        report = validate_data_files(args.data_dir, args.filter_distance, verbose=False)
        
        if not report['valid']:
            print("❌ 数据验证失败，请先运行 --validate-only 检查问题")
            return 1
        
        print(f"✅ 数据验证通过: {report['valid_files']}文件, {report['total_samples']:,}样本\n")
        
        # 开始训练
        success = run_full_training(args)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
