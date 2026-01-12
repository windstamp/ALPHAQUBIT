# 多 NPU 训练指南 (内存优化版)

## 问题背景

原来的多 NPU 训练代码存在以下问题：

1. **内存溢出 (OOM)**: 一次性加载整个数据集到内存
2. **NPU 初始化冲突**: 多个进程同时初始化同一个 NPU
3. **内存泄漏**: 训练完成后没有正确释放内存
4. **进程隔离不足**: 一个任务失败会影响其他任务

## 解决方案

新的 `run_multi_npu_training.py` 脚本包含以下优化：

### 1. 内存优化

| 优化措施 | 描述 |
|---------|------|
| **分块数据加载** | 使用 `mmap_mode='r'` 按需加载数据，不一次性加载全部 |
| **梯度累积** | batch_size=64 × grad_accum=2 = 有效 batch 128 |
| **混合精度训练** | 使用 FP16 减少内存占用 50% |
| **定期清理缓存** | 每 10 个 batch 清理一次 NPU 缓存 |
| **显式垃圾回收** | 每个 epoch 后调用 `gc.collect()` |

### 2. 进程隔离

- 每个 NPU 运行独立的 Python 进程
- 使用 `multiprocessing.spawn` 方法确保干净的进程空间
- 进程之间通过队列通信，互不干扰

### 3. NPU 初始化

```python
# 每个进程只看到一个 NPU
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(npu_id)

# 然后使用 device 0 (映射后的本地设备)
torch.npu.set_device(0)
```

## 使用方法

### 基本用法

```bash
# 使用单个 NPU (调试用)
python run_multi_npu_training.py \
    --data-dir google_finetune_data/finetune \
    --npu-ids 0

# 使用 4 个 NPU
python run_multi_npu_training.py \
    --data-dir google_finetune_data/finetune \
    --npu-ids 0,1,2,3

# 使用全部 8 个 NPU
python run_multi_npu_training.py \
    --data-dir google_finetune_data/finetune \
    --npu-ids 0,1,2,3,4,5,6,7
```

### 内存优化模式

```bash
# 启用所有内存优化 (推荐用于大数据集)
python run_multi_npu_training.py \
    --data-dir google_finetune_data/finetune \
    --npu-ids 0,1,2,3,4,5,6,7 \
    --memory-efficient
```

这会自动设置：
- batch_size = 32
- gradient_accumulation = 4
- num_workers = 1
- gradient_checkpointing = True

### 跳过已完成的实验

```bash
python run_multi_npu_training.py \
    --data-dir google_finetune_data/finetune \
    --npu-ids 0,1,2,3,4,5,6,7 \
    --skip-existing
```

### 指定预训练模型

```bash
python run_multi_npu_training.py \
    --data-dir google_finetune_data/finetune \
    --npu-ids 0,1,2,3,4,5,6,7 \
    --pretrained alphaqubit_pauli_plus.pth
```

## 参数说明

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--data-dir` | (必需) | NPZ 文件目录 |
| `--npu-ids` | `0` | 逗号分隔的 NPU ID 列表 |
| `--batch-size` | `64` | 每个 NPU 的 batch size |
| `--grad-accum` | `2` | 梯度累积步数 |
| `--epochs` | `30` | 最大训练轮数 |
| `--lr` | `1e-5` | 学习率 |
| `--memory-efficient` | False | 启用所有内存优化 |
| `--no-amp` | False | 禁用混合精度 |
| `--skip-existing` | False | 跳过已有模型的实验 |
| `--filter` | None | 按模式过滤实验 |
| `--limit` | None | 限制实验数量 |

## 内存使用估算

每个 NPU (32GB) 的典型内存分配：

| 组件 | 内存 |
|------|------|
| 模型参数 (FP16) | ~100 MB |
| 模型梯度 (FP16) | ~100 MB |
| 优化器状态 | ~400 MB |
| 激活值 (batch=64) | ~2 GB |
| 数据缓冲区 | ~1 GB |
| NPU 系统开销 | ~2 GB |
| **总计** | **~6 GB** |

安全余量：保留 ~26 GB 用于峰值和碎片。

## 故障排除

### 1. NPU 初始化失败

```
RuntimeError: SetPrecisionMode error 500001
```

**解决方案**:
```bash
# 确保环境变量正确
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查 NPU 状态
npu-smi info
```

### 2. 内存溢出

```
RuntimeError: NPU out of memory
```

**解决方案**:
```bash
# 使用更激进的内存优化
python run_multi_npu_training.py \
    --memory-efficient \
    --batch-size 16 \
    --grad-accum 8
```

### 3. 进程卡死

```bash
# 强制终止所有相关进程
pkill -f run_multi_npu_training

# 重置 NPU
npu-smi reset -d 0
```

### 4. 数据加载慢

```bash
# 减少 num_workers 避免内存竞争
python run_multi_npu_training.py --num-workers 1
```

## 与原脚本的对比

| 特性 | 原脚本 | 新脚本 |
|------|--------|--------|
| 数据加载 | 全量加载 | 分块加载 (mmap) |
| 进程模型 | 线程 + subprocess | multiprocessing.spawn |
| 内存管理 | 基本 | 定期清理 + 显式 GC |
| 梯度累积 | 无 | 支持 |
| 混合精度 | CUDA only | NPU + CUDA |
| 错误隔离 | 共享进程 | 独立进程 |

## 预期性能

| 配置 | NPU 数量 | 每实验时间 | 吞吐量 |
|------|---------|-----------|--------|
| 标准 | 1 | ~5 分钟 | 12/小时 |
| 标准 | 4 | ~5 分钟 | 48/小时 |
| 标准 | 8 | ~5 分钟 | 96/小时 |
| 内存优化 | 8 | ~7 分钟 | 68/小时 |

111 个实验预计时间：
- 8 NPU 标准模式: ~70 分钟
- 8 NPU 内存优化: ~100 分钟
