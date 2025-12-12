# AlphaQubit 完整复现指南

本指南详细说明如何在远程NPU服务器上运行完整的AlphaQubit训练流程，并在本地分析结果。

## 目录

1. [概述](#概述)
2. [环境准备](#环境准备)
3. [远程服务器操作](#远程服务器操作)
4. [本地分析操作](#本地分析操作)
5. [文件结构说明](#文件结构说明)
6. [论文对齐验证](#论文对齐验证)

---

## 概述

### 工作流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         远程NPU服务器                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ 数据生成  │ → │  预训练   │ → │   微调   │ → │   测试   │        │
│  │ 8.5M样本 │   │ 100 epochs│   │ 30 epochs│   │ 评估指标 │        │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
│                              ↓                                      │
│                      results/ 文件夹                                │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               │ scp/rsync 下载
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                           本地电脑                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    结果分析与图表生成                          │  │
│  │  - Figure 2: Threshold behavior                              │  │
│  │  - Figure 3: Decoder comparison                              │  │
│  │  - Figure 4: Fine-tuning results                             │  │
│  │  - Extended Data: Ablations                                  │  │
│  │  - Tables: Summary tables                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 论文规格对照

| 类别 | 论文值 | 本实现 |
|------|--------|--------|
| **预训练样本** | 8.5M | ✅ 8,500,000 |
| **预训练 Epochs** | 100 | ✅ 100 |
| **预训练 Batch Size** | 256 | ✅ 256 |
| **预训练 LR** | 1e-4 | ✅ 1e-4 |
| **微调样本/实验** | 50K | ✅ 50,000 |
| **微调 Epochs** | 30 | ✅ 30 |
| **微调 Batch Size** | 128 | ✅ 128 |
| **微调 LR** | 1e-5 | ✅ 1e-5 |
| **模型架构** | 256d/8h/12L | ✅ Large config |
| **噪声模型** | Pauli+ (Table S4) | ✅ 完全对齐 |

---

## 环境准备

### 远程服务器要求

```bash
# 硬件要求
- NPU/GPU: 华为Ascend NPU 或 NVIDIA GPU (推荐 16GB+ 显存)
- 内存: 64GB+ RAM
- 存储: 100GB+ 可用空间

# 软件要求
- Python 3.9+
- PyTorch 2.0+ (with NPU/CUDA support)
- stim
- numpy, scipy, matplotlib
```

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/xuda1979/ALPHAQUBIT.git
cd ALPHAQUBIT

# 2. 创建conda环境
conda create -n alphaqubit python=3.10
conda activate alphaqubit

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import stim; import torch; print('OK')"
```

---

## 远程服务器操作

### 快速测试 (推荐首次运行)

```bash
# 快速测试模式 - 验证流程正确性
python run_server_pipeline.py --quick-test --output-dir ./test_run

# 预计时间: 10-30分钟
# 输出: test_run/ 目录
```

### 完整运行 (论文复现)

```bash
# 完整运行 - 与论文完全一致
python run_server_pipeline.py \
    --output-dir ./results_full \
    --device npu \
    --pretrain-samples 8500000 \
    --pretrain-epochs 100 \
    --finetune-epochs 30

# 预计时间: 24-72小时 (取决于硬件)
```

### 自定义运行

```bash
# 自定义参数
python run_server_pipeline.py \
    --output-dir ./results_custom \
    --device cuda \
    --pretrain-samples 1000000 \
    --pretrain-epochs 50 \
    --finetune-epochs 15
```

### 监控运行状态

```bash
# 查看日志
tail -f results_*/pipeline.log

# 查看进度
cat results_*/pipeline_metrics.json | python -m json.tool
```

---

## 本地分析操作

### 下载结果

```bash
# 从服务器下载结果
scp -r user@server:/path/to/results_full ./results_from_server

# 或使用rsync (增量同步)
rsync -avz user@server:/path/to/results_full ./results_from_server
```

### 运行分析

```bash
# 基本分析
python analyze_server_results.py --results-dir ./results_from_server

# 生成PDF版本
python analyze_server_results.py --results-dir ./results_from_server --pdf

# 指定输出目录
python analyze_server_results.py \
    --results-dir ./results_from_server \
    --output-dir ./my_analysis
```

### 生成的图表

分析完成后，在 `output_dir/` 中会生成以下文件：

| 文件 | 描述 | 对应论文 |
|------|------|----------|
| `fig2_threshold_analysis.png` | 阈值行为图 | Figure 2 |
| `fig3_decoder_comparison.png` | 解码器对比图 | Figure 3 |
| `fig4_finetuning_results.png` | 微调结果图 | Figure 4 |
| `extended_ablations.png` | 消融实验图 | Extended Data |
| `training_curves.png` | 训练曲线 | 补充材料 |
| `table_results_summary.txt` | 结果汇总表 | - |
| `table_paper_comparison.txt` | 与论文对比 | - |
| `table_results.tex` | LaTeX表格 | - |
| `ANALYSIS_REPORT.md` | 分析报告 | - |

---

## 文件结构说明

### 结果目录结构

```
results_YYYYMMDD_HHMMSS/
├── config.json              # 运行配置
├── pipeline.log             # 完整日志
├── pipeline_metrics.json    # 性能指标
│
├── data/                    # 生成的数据
│   ├── dem/                 # DEM噪声数据
│   │   ├── dem_d3_r01.npz
│   │   ├── dem_d3_r05.npz
│   │   └── ...
│   ├── si1000/              # SI1000噪声数据
│   │   ├── si1000_d3_r01_p0p001.npz
│   │   └── ...
│   ├── pauli_plus/          # Pauli+微调数据
│   │   ├── samples_surface_code_bX_d3_r01.npz
│   │   └── ...
│   └── test/                # 测试数据
│
├── pretrain/                # 预训练结果
│   ├── pretrained_model.pth # 预训练模型
│   └── train_history.json   # 训练历史
│
├── finetune/                # 微调结果
│   ├── samples_surface_code_bX_d3_r01/
│   │   ├── finetuned_model.pth
│   │   └── finetune_history.json
│   └── ...
│
├── test/                    # 测试结果
│   ├── test_results.json    # 所有测试指标
│   └── predictions_*.npz    # 预测结果
│
└── figures/                 # 生成的图表
    ├── fig2_threshold.png
    ├── fig3_decoder_comparison.png
    ├── fig4_finetuning.png
    └── results_summary.csv
```

### 关键文件格式

#### config.json
```json
{
  "pretrain_samples_total": 8500000,
  "code_distances": [3, 5, 7],
  "hidden_dim": 256,
  "num_heads": 8,
  "num_layers": 12,
  "pretrain_epochs": 100,
  "finetune_epochs": 30
}
```

#### test_results.json
```json
{
  "results": [
    {
      "model": "samples_surface_code_bX_d3_r01",
      "test": "test_bX_d3_r01",
      "ler": 0.0285,
      "accuracy": 0.9715,
      "num_samples": 10000
    }
  ],
  "time_seconds": 123.45
}
```

---

## 论文对齐验证

### 运行验证测试

```bash
# 运行39项自动验证
python -m verification.verify_paper_alignment

# 预期输出: 39/39 tests pass (100%)
```

### 验证内容

1. **Table S4 噪声参数** (10项)
   - cycle_ns, T1_us, Tphi_us
   - p_readout, p_reset, p_heat_12
   - p_cz_leak_11_to_02, p_cz_crosstalk_ZZ
   - p_1q_excess, p_cz_excess

2. **模型架构** (3项)
   - Hidden dimension: 256
   - Number of heads: 8
   - Number of layers: 12

3. **Kraus算符** (7项)
   - 所有通道满足CPTP条件

4. **训练超参数** (4项)
   - BCEWithLogitsLoss
   - AdamW optimizer
   - OneCycleLR scheduler
   - Gradient clipping

5. **其他** (15项)
   - Soft XOR公式
   - GPTA实现
   - DQLR矩阵
   - 位置编码

---

## 常见问题

### Q: 预训练太慢怎么办？

A: 可以减少样本数进行初步实验：
```bash
python run_server_pipeline.py --pretrain-samples 1000000 --pretrain-epochs 50
```

### Q: 内存不足怎么办？

A: 减小batch size:
```bash
# 修改 run_server_pipeline.py 中的配置
pretrain_batch_size: int = 128  # 从256减少到128
```

### Q: 如何只运行部分阶段？

A: 目前需要修改代码。未来版本将支持：
```bash
python run_server_pipeline.py --stage pretrain  # 只预训练
python run_server_pipeline.py --stage finetune  # 只微调
```

### Q: 如何使用已有的预训练模型？

A: 将预训练模型放入正确位置：
```bash
cp my_pretrained.pth results_dir/pretrain/pretrained_model.pth
```

---

## 联系方式

如有问题，请提交 GitHub Issue 或联系项目维护者。

---

*最后更新: 2024-12-10*
