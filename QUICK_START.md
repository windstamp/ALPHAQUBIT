# AlphaQubit 一键运行指南

## 🚀 快速开始

### 第一步：诊断当前结果

```bash
cd ~/work/ALPHAQUBIT
python3 diagnose_current_results.py
```

这会分析你当前的测试结果，指出问题所在。

### 第二步：运行完整改进流程

```bash
# 完整流程 (需要1-3天，取决于数据量)
python3 run_full_pipeline.py

# 或者快速测试模式 (2-4小时)
python3 run_full_pipeline.py --quick-test

# 或者仅评估现有模型
python3 run_full_pipeline.py --only-evaluate
```

## 📋 流程说明

`run_full_pipeline.py` 会自动执行以下6个步骤：

1. **生成预训练数据** - 覆盖0.1%-25%噪声范围
2. **预训练基础模型** - 在扩展数据上训练
3. **Fine-tune所有实验** - 118个实验配置
4. **评估模型** - 在测试集上评估
5. **生成对比图表** - 与论文baseline对比
6. **诊断分析** - 找出剩余问题

## ⚙️ 选项说明

| 选项 | 说明 | 适用场景 |
|-----|------|---------|
| `--quick-test` | 快速测试模式 | 验证流程是否正常 |
| `--skip-pretrain` | 跳过预训练 | 已有预训练模型 |
| `--skip-finetune` | 跳过fine-tuning | 已有fine-tuned模型 |
| `--only-evaluate` | 仅评估 | 只想看结果 |

## 📊 查看结果

运行完成后，查看以下文件：

```bash
# 可视化对比图
test_results_v2/ler_comparison_with_paper.png

# 详细分析
cat test_results_v2/paper_comparison_analysis.json | python3 -m json.tool

# 流程日志
tail -f pipeline_log.txt
```

## 🔍 故障排查

### 如果某个步骤失败

1. 查看 `pipeline_log.txt` 找到错误信息
2. 查看 `pipeline_results.json` 看哪一步失败
3. 单独运行失败的步骤进行调试

### 如果数据生成失败

```bash
# 手动生成单个噪声级别
python3 generate_data.py --model pauli_plus --samples 50000
```

### 如果训练中断

```bash
# 从fine-tuning开始继续
python3 run_full_pipeline.py --skip-pretrain

# 或从评估开始
python3 run_full_pipeline.py --only-evaluate
```

## 📈 预期改善

运行完整流程后，预期性能：

| 指标 | 改进前 | 改进后 |
|-----|--------|--------|
| 平均LER | 20.13% | **5-8%** |
| 最好模型 | 3.74% | **2.5-3%** |
| 最差模型 | 25.36% | **8-12%** |

## ⏱️ 预计时间

| 模式 | 时间 | 说明 |
|-----|------|-----|
| `--quick-test` | 2-4小时 | 验证流程 |
| 标准模式 | 1-2天 | 生产环境 |
| `--only-evaluate` | 30分钟 | 仅评估 |

## 💡 提示

1. **首次运行建议**: 先用 `--quick-test` 验证流程
2. **生产运行**: 去掉 `--quick-test`，完整训练
3. **断点续传**: 使用 `--skip-*` 选项跳过已完成的步骤
4. **监控训练**: 用 `tail -f pipeline_log.txt` 实时查看

## 🆘 获取帮助

```bash
python3 run_full_pipeline.py --help
python3 diagnose_current_results.py --help
```

## 📁 输出文件

```
test_results_v2/
├── ler_comparison_with_paper.png      # 对比可视化
├── paper_comparison_analysis.json     # 详细分析
├── test_summary.json                  # 原始结果
└── predictions/                       # 预测文件

pipeline_log.txt                       # 执行日志
pipeline_results.json                  # 流程结果
```
