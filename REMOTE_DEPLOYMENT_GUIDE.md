# AlphaQubit 远程服务器部署与本地分析指南

## 🚀 快速开始

### 1️⃣ 本地设置（Windows）

```powershell
# 在 ALPHAQUBIT 目录下运行
cd c:\Users\Lenovo\software\ALPHAQUBIT
.\local_setup\setup_local.ps1
```

这会：
- 配置 SSH 连接
- 创建便捷脚本
- 指导安装 VSCode Remote-SSH

### 2️⃣ 上传代码到服务器

```powershell
.\local_scripts\upload_code.ps1
```

### 3️⃣ 在服务器上部署环境

```powershell
# 连接服务器
.\local_scripts\connect_server.ps1

# 在服务器上运行
chmod +x ~/alphaqubit/ALPHAQUBIT/remote_scripts/*.sh
~/alphaqubit/ALPHAQUBIT/remote_scripts/setup_remote_env.sh

# 安装可选依赖（MWPM解码器）
pip install pymatching stim
```

### 4️⃣ 运行解码器基准测试（生成论文对比图）

```bash
# 快速测试（验证环境）
python run_npu_benchmark.py --test

# 完整基准测试（全部距离、全部错误率）
python run_npu_benchmark.py --full --npu

# 使用训练好的模型
python run_npu_benchmark.py --full --npu --model alphaqubit_pauli_plus.pth
```

### 5️⃣ 运行训练

```bash
# 快速测试（验证环境）
~/alphaqubit/run_quick_test.sh

# 完整训练（后台运行）
~/alphaqubit/run_full_training.sh
```

### 6️⃣ 下载并分析结果

```powershell
# 本地运行
.\local_scripts\download_results.ps1
.\local_scripts\analyze_results.ps1
```

---

## 📊 生成论文图表

### 方法1：使用NPU基准测试脚本（推荐）

```bash
# 在远程服务器上运行
python run_npu_benchmark.py --full --npu --paper-data

# 输出目录结构：
# npu_benchmark_results/benchmark_YYYYMMDD_HHMMSS/
# ├── benchmark_results.json
# ├── BENCHMARK_REPORT.md
# └── figures/
#     ├── fig2_threshold_comparison.png  # Figure 2: 阈值对比
#     ├── fig3_decoder_comparison_benchmark.png  # Figure 3: 解码器对比
#     └── improvement_summary.png  # 改进总结
```

### 方法2：使用通用基准测试脚本

```bash
python run_decoder_benchmark.py --full --device npu

# 输出目录：benchmark_results/
```

### 方法3：生成完整研究报告

```bash
python generate_research_report_complete.py

# 输出：research_report/paper_aligned_report.md
```

---

## 详细步骤

### Step 1: 配置 SSH 连接

#### 方法A: 使用设置脚本（推荐）

```powershell
.\local_setup\setup_local.ps1
```

#### 方法B: 手动配置

1. 编辑 `C:\Users\<用户名>\.ssh\config`：

```
Host cmcc-npu
    HostName <服务器IP>
    User <用户名>
    Port 22
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

2. 测试连接：
```powershell
ssh cmcc-npu
```

---

### Step 2: 配置 VSCode Remote-SSH

1. **安装扩展**
   - 打开 VSCode
   - 按 `Ctrl+Shift+X`
   - 搜索 `Remote - SSH`
   - 点击安装

2. **连接到服务器**
   - 按 `F1`
   - 输入 `Remote-SSH: Connect to Host`
   - 选择 `cmcc-npu`
   - 输入密码（如果需要）

3. **打开远程文件夹**
   - 连接成功后，点击 `打开文件夹`
   - 选择 `~/alphaqubit/ALPHAQUBIT`

---

### Step 3: 上传代码到服务器

```powershell
# 方法1: 使用脚本
.\local_scripts\upload_code.ps1

# 方法2: 手动上传
scp -r C:\Users\Lenovo\software\ALPHAQUBIT cmcc-npu:~/alphaqubit/
```

---

### Step 4: 服务器环境部署

SSH 连接到服务器后运行：

```bash
# 设置脚本执行权限
chmod +x ~/alphaqubit/ALPHAQUBIT/remote_scripts/*.sh

# 运行部署脚本
~/alphaqubit/ALPHAQUBIT/remote_scripts/setup_remote_env.sh
```

部署脚本会自动：
- 检查 NPU 环境
- 安装 Miniconda（如果需要）
- 创建 conda 环境
- 安装所有依赖
- 验证安装
- 创建运行脚本

---

### Step 5: 运行训练

#### 快速测试（建议先运行）

```bash
~/alphaqubit/run_quick_test.sh
```

- 用时：约 10-30 分钟
- 目的：验证环境和流程正确

#### 完整训练

```bash
~/alphaqubit/run_full_training.sh
```

- 用时：24-72 小时
- 后台运行，SSH 断开不影响
- 自动保存日志

#### 监控训练

```bash
# 查看实时日志
tail -f ~/alphaqubit/ALPHAQUBIT/training_*.log

# 查看进度
grep -E "Stage|完成|Epoch" ~/alphaqubit/ALPHAQUBIT/training_*.log

# 检查进程
ps aux | grep python

# 检查 NPU 使用情况
npu-smi info
```

---

### Step 6: 下载结果

```powershell
# 方法1: 使用脚本
.\local_scripts\download_results.ps1

# 方法2: 手动下载
scp -r cmcc-npu:~/alphaqubit/ALPHAQUBIT/results_* ./downloaded_results/
```

---

### Step 7: 本地分析

```powershell
# 使用脚本
.\local_scripts\analyze_results.ps1

# 或手动运行
python analyze_server_results.py --results-dir ./downloaded_results/results_xxx --output-dir ./analysis --pdf
```

生成的文件：
- `fig2_threshold_analysis.png/pdf` - 阈值行为图
- `fig3_decoder_comparison.png/pdf` - 解码器对比
- `fig4_finetuning_results.png/pdf` - 微调结果
- `extended_ablations.png/pdf` - 消融实验
- `table_results_summary.txt` - 结果汇总
- `ANALYSIS_REPORT.md` - 分析报告

---

## 文件结构

```
ALPHAQUBIT/
├── remote_scripts/           # 远程服务器脚本
│   ├── setup_remote_env.sh   # 环境部署脚本
│   ├── run_full_training.sh  # 完整训练脚本
│   ├── run_quick_test.sh     # 快速测试脚本
│   └── pack_results.sh       # 打包结果脚本
│
├── local_setup/              # 本地设置
│   ├── setup_local.ps1       # 本地设置脚本
│   └── ssh_config_template   # SSH配置模板
│
├── local_scripts/            # 本地便捷脚本 (运行setup_local.ps1后生成)
│   ├── connect_server.ps1
│   ├── upload_code.ps1
│   ├── download_results.ps1
│   ├── view_log.ps1
│   └── analyze_results.ps1
│
├── run_server_pipeline.py    # 完整训练流水线
├── analyze_server_results.py # 结果分析脚本
└── ...
```

---

## 常见问题

### Q: SSH 连接被拒绝

检查：
1. 服务器 IP 和端口是否正确
2. 用户名是否正确
3. 防火墙是否允许 SSH

### Q: SSH 连接频繁断开

在 `~/.ssh/config` 中添加：
```
ServerAliveInterval 60
ServerAliveCountMax 3
```

### Q: 训练中断后如何恢复

目前需要重新开始。建议使用 `nohup` 和 `screen`：
```bash
screen -S alphaqubit
~/alphaqubit/run_full_training.sh
# 按 Ctrl+A, D 分离
# screen -r alphaqubit 恢复
```

### Q: NPU 不可用

检查：
1. 运行 `npu-smi info` 查看状态
2. 确保 CANN toolkit 已安装
3. 确保 torch_npu 已安装

### Q: 内存不足

减小 batch size：编辑 `run_server_pipeline.py` 中的配置。

---

## 联系与支持

如有问题，请提交 GitHub Issue。
