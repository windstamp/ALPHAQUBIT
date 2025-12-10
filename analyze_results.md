# 评估结果分析 - 与AlphaQubit论文对比

## 整体结果

- **平均 LER**: 20.13% (论文baseline: 3%)
- **最好模型**: surface_code_bX_d3_r01_center_5_7 → 3.74% LER ✓ 接近论文
- **最差模型**: surface_code_bX_d5_r25_center_5_5 → 25.36% LER
- **总实验数**: 118个模型

## 关键发现

### 1. 表现良好的配置模式
Top 10 模型都具有以下特征：
- **Code distance = 3** (d3)
- **Noise rate = 0.01** (r01)
- **Code type**: surface_code (bX 和 bZ)

这说明：
- ✓ 在低噪声、小距离下，模型能够接近论文性能
- ✗ 在高噪声、大距离下性能急剧下降

### 2. 可能的问题根源

#### A. 训练数据问题
- [ ] **数据分布不匹配**: 预训练数据噪声范围是否覆盖测试范围？
  - 检查: `make_all_pretraining_noise.py` 生成的噪声参数
  - 对比: 测试数据的噪声率 (r01=1%, r25=25%)
  
- [ ] **训练样本数量不足**: 高噪声/大距离场景的样本是否足够？
  - 检查各个实验的训练集大小
  - 是否需要增加困难样本的比例

#### B. Fine-tuning策略问题
- [ ] **学习率过高/过低**: 导致catastrophic forgetting或训练不充分
  - 检查: `ai_models/fine_tune_npz.py` 中的学习率设置
  
- [ ] **训练轮数不足**: 特别是困难场景下
  - 检查日志: 各实验是否都收敛了？
  
- [ ] **权重衰减/正则化**: 是否过拟合/欠拟合

#### C. 模型架构问题
- [ ] **预训练模型质量**: 基础模型是否足够好？
  - 验证预训练模型在验证集上的表现
  
- [ ] **模型容量**: 对于 d5、d7 这样的大码，hidden_dim=256 是否够用？
  - 论文可能使用了更大的模型

#### D. 数据标注/格式问题
- [ ] **标签错误**: prediction文件和真实标签是否对齐？
  - 验证: 随机抽查几个样本，手动检查预测和标签
  
- [ ] **特征提取**: detector readings 是否正确提取？

### 3. 立即可以做的检查

#### 在服务器上运行以下命令：

```bash
cd ~/work/ALPHAQUBIT

# 1. 检查训练日志，看看模型是否收敛
ls -lh finetuned_models/*.log 2>/dev/null || echo "No log files"

# 2. 看看最好和最差模型的训练配置差异
# (如果有训练日志的话)

# 3. 检查一个简单case的预测分布
python3 -c "
import numpy as np
# 检查最好的模型
data = np.load('test_results/predictions/surface_code_bX_d3_r01_center_5_7_predictions.npz')
preds = data['predictions']
print(f'Best model - Prediction distribution: 0={np.sum(preds==0)}, 1={np.sum(preds==1)}')

# 检查最差的模型  
data = np.load('test_results/predictions/surface_code_bX_d5_r25_center_5_5_predictions.npz')
preds = data['predictions']
print(f'Worst model - Prediction distribution: 0={np.sum(preds==0)}, 1={np.sum(preds==1)}')

# 检查对应的真实标签分布
test_data = np.load('google_finetune_data/test/samples_surface_code_bX_d3_r01_center_5_7.npz')
labels = test_data['y']
print(f'Best model labels - Label distribution: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}')

test_data = np.load('google_finetune_data/test/samples_surface_code_bX_d5_r25_center_5_5.npz')
labels = test_data['y']
print(f'Worst model labels - Label distribution: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}')
"

# 4. 检查预训练数据的噪声范围
python3 -c "
import yaml
with open('configs/pauli_plus.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print('Pretraining noise config:')
    print(config)
"
```

## 下一步行动计划

### 短期（诊断）
1. ✓ 已生成完整评估报告
2. 下载并查看可视化图: `test_results/ler_comparison_with_paper.png`
3. 运行上述诊断命令，确认数据分布
4. 检查fine-tuning超参数配置

### 中期（改进）
根据诊断结果：
- **如果是数据问题**: 重新生成训练数据，增加困难样本
- **如果是训练问题**: 调整学习率、训练轮数、batch size
- **如果是模型问题**: 增加模型容量或改进架构
- **如果是标签问题**: 修复标注流程

### 长期（优化）
- 实现课程学习：从简单(d3, r01)到困难(d7, r25)
- 增加数据增强策略
- 尝试集成学习或蒸馏

## 重要文件位置

### 当前结果
- 评估摘要: `test_results/test_summary.json`
- 对比分析: `test_results/paper_comparison_analysis.json`
- 可视化: `test_results/ler_comparison_with_paper.png`
- 预测文件: `test_results/predictions/*.npz`

### 需要检查的代码
- Fine-tuning入口: `run_finetune_all.py` 或 `run_fine_tune_all.py`
- Fine-tuning逻辑: `ai_models/fine_tune_npz.py`
- 数据生成: `make_all_pretraining_noise.py`
- 模型定义: `ai_models/model_mla.py`
- 配置文件: `configs/pauli_plus.yaml`

### 训练数据位置（需确认）
- 预训练数据: `pretraining_data/` 或 `simulated_data/`
- Fine-tuning数据: `google_finetune_data/train/`
- 测试数据: `google_finetune_data/test/`

## 论文关键细节对照（需要查阅）

查看论文 `s41586-024-08449-y.pdf` 或 `41586_2024_8449_MOESM1_ESM.pdf`:
- [ ] 确认论文使用的预训练数据规模和噪声范围
- [ ] 确认fine-tuning的超参数（learning rate, epochs, batch size）
- [ ] 确认模型架构细节（hidden_dim, num_heads, num_layers）
- [ ] 确认测试集的构成（是否平衡？各噪声率的比例？）
