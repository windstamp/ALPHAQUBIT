# AlphaQubit 本地编辑 + 远程训练 工作流指南

## 📋 概述

本指南介绍如何实现 **本地编辑代码，远程服务器训练** 的工作流程，使用 S3 作为中转站。

```
┌─────────────────┐         ┌───────────────┐         ┌─────────────────┐
│   本地 Windows   │         │      S3       │         │   远程服务器     │
│                 │  上传   │               │   拉取   │                 │
│  1. 编辑代码     │ ──────► │  ALPHAQUBIT/  │ ◄────── │  2. 拉取代码    │
│                 │         │               │         │  3. NPU训练     │
│  5. 下载结果     │ ◄────── │   results/    │ ──────► │  4. 上传结果    │
│  6. 分析结果     │         │               │         │                 │
└─────────────────┘         └───────────────┘         └─────────────────┘
```

---

## 🔧 前置要求

### 本地 Windows
- Python 3.8+
- rclone（已配置好 `nm-aihuanxin` remote）
- watchdog 库：`pip install watchdog`

### 远程服务器
- rclone（已配置好 `nm-aihuanxin` remote）
- Python 环境 + PyTorch NPU

### 验证 rclone 配置
```bash
# 本地或服务器运行
rclone lsd nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/
```

---

## 📁 相关脚本文件

| 文件 | 位置 | 用途 |
|------|------|------|
| `auto_sync_to_s3.py` | 本地根目录 | 监控文件变化，自动上传到 S3 |
| `train_with_s3_sync.sh` | `remote_scripts/` | 服务器一键训练脚本 |
| `auto_sync_from_s3.sh` | `remote_scripts/` | 服务器持续同步脚本 |

---

## 🚀 使用流程

### 步骤 1：首次上传代码到 S3（本地）

```powershell
cd C:\Users\Lenovo\software\ALPHAQUBIT
rclone copy . nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT --progress --transfers 8
```

### 步骤 2：启动本地自动同步（本地）

打开一个 PowerShell 窗口，运行：

```powershell
cd C:\Users\Lenovo\software\ALPHAQUBIT
python auto_sync_to_s3.py
```

**保持此窗口运行**。现在你编辑任何文件，5秒后会自动上传到 S3。

### 步骤 3：服务器首次设置（服务器）

```bash
# 从 S3 拉取代码
mkdir -p /root/work
rclone sync nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT /root/work/ALPHAQUBIT --progress

# 设置脚本权限
chmod +x /root/work/ALPHAQUBIT/remote_scripts/*.sh

# 验证 NPU
python -c "import torch; import torch_npu; print('NPU:', torch.npu.is_available())"
```

### 步骤 4：运行训练（服务器）

#### 方法 A：一键训练（推荐）

```bash
cd /root/work/ALPHAQUBIT
./remote_scripts/train_with_s3_sync.sh
```

这会自动：
1. 从 S3 拉取最新代码
2. 运行完整训练
3. 上传结果到 S3

#### 方法 B：快速测试

```bash
./remote_scripts/train_with_s3_sync.sh --quick
```

#### 方法 C：手动运行

```bash
cd /root/work/ALPHAQUBIT

# 1. 拉取最新代码
rclone sync nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT . \
    --exclude "pretrain_data/**" --exclude "backup_*/**" --progress

# 2. 运行训练
python run_full_pipeline.py --npu

# 3. 上传结果
rclone copy test_results_v2 nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/test_results_v2 --progress
rclone copy pipeline_results.json nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/ --progress
```

### 步骤 5：下载结果（本地）

```powershell
cd C:\Users\Lenovo\software\ALPHAQUBIT

# 下载测试结果
rclone copy nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/test_results_v2 ./test_results_v2 --progress

# 下载训练好的模型
rclone copy nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/finetuned_models_v2 ./finetuned_models_v2 --progress

# 下载日志
rclone copy nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/pipeline_results.json ./ --progress
```

### 步骤 6：分析结果（本地）

```powershell
python analyze_server_results.py
python plot_alphaquibit_results.py
```

---

## 📋 日常工作流程（简化版）

### 每天开始工作时

**本地**（开一个终端保持运行）：
```powershell
cd C:\Users\Lenovo\software\ALPHAQUBIT
python auto_sync_to_s3.py
```

### 编辑代码

正常在 VS Code 中编辑文件，保存后自动上传。

### 需要训练时

**服务器**：
```bash
cd /root/work/ALPHAQUBIT
./remote_scripts/train_with_s3_sync.sh --quick  # 或不加 --quick 完整训练
```

### 查看结果

**本地**：
```powershell
rclone copy nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/test_results_v2 ./test_results_v2 --progress
```

---

## 🔧 高级用法

### 服务器持续监控模式

如果想让服务器自动同步代码（每60秒检查一次）：

```bash
# 后台运行
nohup /root/work/ALPHAQUBIT/remote_scripts/train_with_s3_sync.sh --watch > /tmp/sync.log 2>&1 &

# 查看日志
tail -f /tmp/sync.log

# 停止
pkill -f "train_with_s3_sync.sh"
```

### 后台运行训练

```bash
nohup ./remote_scripts/train_with_s3_sync.sh > training.log 2>&1 &

# 查看进度
tail -f training.log
```

### 监控 NPU 使用

```bash
watch -n 2 npu-smi info
```

---

## ⚠️ 注意事项

1. **不要同时在本地和服务器编辑同一个文件**，可能会造成冲突
2. **大文件（.npy, pretrain_data/）不会同步**，节省带宽
3. **训练数据在服务器本地生成**，不通过 S3 传输
4. **保持 `auto_sync_to_s3.py` 运行**，否则本地修改不会上传

---

## 🔍 故障排除

### rclone 连接失败
```bash
# 检查配置
rclone config show nm-aihuanxin

# 测试连接
rclone lsd nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/
```

### NPU 未使用
```bash
# 检查 NPU
python -c "import torch; import torch_npu; print(torch.npu.is_available())"

# 查看 NPU 状态
npu-smi info
```

### 同步没有生效
```bash
# 手动同步测试
rclone copy /root/work/ALPHAQUBIT/test.txt nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/ -v
```

---

## 📞 快速参考命令

| 操作 | 本地命令 | 服务器命令 |
|------|----------|------------|
| 上传代码 | `rclone copy . nm-aihuanxin:.../ALPHAQUBIT --progress` | - |
| 下载代码 | - | `rclone sync nm-aihuanxin:.../ALPHAQUBIT . --progress` |
| 自动同步 | `python auto_sync_to_s3.py` | `./remote_scripts/train_with_s3_sync.sh --watch` |
| 快速训练 | - | `./remote_scripts/train_with_s3_sync.sh --quick` |
| 完整训练 | - | `./remote_scripts/train_with_s3_sync.sh` |
| 下载结果 | `rclone copy nm-aihuanxin:.../ALPHAQUBIT/test_results_v2 ./` | - |

---

## 📝 S3 路径

```
nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/
├── ai_models/           # 模型代码
├── remote_scripts/      # 远程脚本
├── pretrained_models/   # 预训练模型（训练后生成）
├── finetuned_models_v2/ # 微调模型（训练后生成）
├── test_results_v2/     # 测试结果（训练后生成）
├── pipeline_results.json
├── pipeline_log.txt
└── ...
```
