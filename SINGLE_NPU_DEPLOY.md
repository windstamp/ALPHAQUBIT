# 部署单卡 NPU 训练脚本

## 问题摘要

之前的训练在分布式 8-NPU 模式下运行约 1.5 小时后崩溃，出现 HCCL 通信超时错误：
```
Cluster Exception Location[IP/ID]:[10.222.9.255/2], ExceptionType:[Stuck Occurred]
error code is 507048... The execution of the internal task times out
```

## 解决方案

创建单卡训练脚本 `train_single_npu.py`，避免分布式训练的复杂性，先验证代码正确性。

## 更新说明

1. **新增 `train_single_npu.py`** - 单卡 NPU 训练脚本
2. **改进 `rl_algorithms/self_play.py`** - 更频繁的日志记录（每 10 步）
3. **新增 `comprehensive_test.py`** - 综合本地测试脚本

## 部署命令 (PowerShell)

```powershell
# 1. 进入项目目录
cd "c:\Users\Lenovo\software\quantum-qwen25-coder-main"

# 2. 上传所有更新文件到 S3
New-Item -ItemType Directory -Force -Path upload_temp
New-Item -ItemType Directory -Force -Path upload_temp\rl_algorithms

Copy-Item "train_single_npu.py" "upload_temp\"
Copy-Item "comprehensive_test.py" "upload_temp\"
Copy-Item "rl_algorithms\self_play.py" "upload_temp\rl_algorithms\"
Copy-Item "rl_algorithms\__init__.py" "upload_temp\rl_algorithms\"

rclone copy upload_temp nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3 -v

Remove-Item -Recurse -Force upload_temp
```

## 服务器端命令

```bash
# 1. 进入项目目录
cd /root/work/david/software/quantum-qwen25-coder-main

# 2. 从 S3 拉取更新
rclone copy nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/train_single_npu.py . -v
rclone copy nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/comprehensive_test.py . -v
rclone copy nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/rl_algorithms ./rl_algorithms -v

# 3. 停止任何正在运行的 Python 进程
pkill -9 -f python; sleep 3

# 4. 运行单卡训练 (device_id 选择空闲的 NPU)
python train_single_npu.py \
    --model_name_or_path models/Qwen2.5-Coder-0.5B-Instruct \
    --output_dir outputs/quantum_single_$(date +%Y%m%d_%H%M%S) \
    --num_epochs 3 \
    --problems_per_epoch 30 \
    --device_id 0 \
    2>&1 | tee training_single.log
```

## 日志输出样例

训练过程中会看到类似以下的日志（每 10 步输出一次）：
```
[Epoch 1/3] Step 10/30, Loss: 0.1234, Avg Reward: 0.350, Best Reward: 1.00, Difficulty: 1.0
[Epoch 1/3] Step 20/30, Loss: 0.0987, Avg Reward: 0.450, Best Reward: 1.00, Difficulty: 1.0
[Epoch 1/3] Step 30/30, Loss: 0.0765, Avg Reward: 0.520, Best Reward: 1.00, Difficulty: 1.0
Epoch 1 completed. Avg Reward: 0.520, Correct: 15/120
```

## 如果单卡训练成功

单卡训练验证代码正确后，可以考虑：
1. 逐步增加 NPU 数量（2卡、4卡、8卡）
2. 使用较小的 batch_size 减少内存压力
3. 调整 HCCL 超时时间

## 如果单卡训练仍有问题

请提供完整的错误日志，包括：
1. Python 版本
2. torch_npu 版本
3. 完整的 traceback
