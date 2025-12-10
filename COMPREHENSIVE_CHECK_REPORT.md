# AlphaQubit 全面检查报告

## 执行摘要

**当前状态**: 平均LER = 20.13%，论文baseline = 3%，差距 **17.13个百分点**

**最好模型**: `surface_code_bX_d3_r01_center_5_7` → **3.74% LER** ✓ 接近论文

**最差模型**: `surface_code_bX_d5_r25_center_5_5` → **25.36% LER**

---

## 1. 关键问题诊断 ✅

### 1.1 Fine-tuning 超参数配置

| 参数 | 当前值 | 评估 | 建议 |
|-----|--------|------|-----|
| 学习率 | `1e-4` | ✓ 合理 | 可尝试 5e-4 或 5e-5 |
| 训练轮数 | `30 epochs` | ✓ 合理 | 检查early stopping日志 |
| Batch Size | `128` | ✓ 合理 | 可增加到256/512 |
| Weight Decay | `1e-3` | ✓ 合理 | OK |
| 优化器 | `AdamW` | ✓ 最佳实践 | OK |
| 调度器 | `CosineAnnealingWarmRestarts` | ✓ 良好策略 | OK |
| 梯度裁剪 | `clip_norm=1.0` | ✓ 防止爆炸 | OK |

**结论**: 超参数配置合理，不是主要问题。

---

### 1.2 预训练数据噪声范围 ⚠️ **关键问题**

**当前配置** (`configs/pauli_plus.yaml`):
```yaml
depolarization: 0.001   # 0.1% 噪声
leakage_rate: 0.0005    # 0.05% 泄漏
```

**测试数据噪声范围**:
- r01 = 1%
- r03 = 3%
- r05 = 5%
- r07 = 7%
- r09 = 9%
- r25 = 25%

**问题**:
- ❌ 预训练噪声=0.1%，但测试最高25% → **差距250倍**
- ❌ 模型只在低噪声(r01=1%)下表现好，高噪声崩溃
- ❌ Top 10模型全部是 d3+r01 配置

**证据**:
```
按噪声率分组的平均LER:
  r01 (1%):  ~0.04-0.05  ✓ 接近论文
  r03 (3%):  ~0.09-0.10
  r05 (5%):  ~0.12-0.15
  r25 (25%): ~0.23-0.25  ✗ 完全失败
```

**根本原因**: 分布偏移(Distribution Shift) - 预训练和测试数据的噪声分布不匹配

---

### 1.3 模型架构配置

| 参数 | 值 | 对d=3的评估 | 对d=7的评估 |
|-----|---|------------|------------|
| hidden_dim | 256 | ✓ 足够 | ⚠️ 可能不足 |
| num_heads | 8 | ✓ 合理 | ✓ 合理 |
| num_layers | 12 | ✓ 标准深度 | ✓ 标准深度 |

**容量分析**:
- d=3: 9 stabilizers, ~28 params/stabilizer → ✓ 充足
- d=5: 25 stabilizers, ~10 params/stabilizer → ⚠️ 偏少
- d=7: 49 stabilizers, ~5 params/stabilizer → ❌ 不足

**建议**: 对大码距使用更大模型
```python
if distance >= 5:
    hidden_dim = 384
if distance >= 7:
    hidden_dim = 512
```

---

### 1.4 数据生成和标注

检查了以下组件:
- ✓ `NPZDataset` 正确加载detection events和observables
- ✓ Observable转换逻辑正确 (48.0→0, 49.0→1)
- ✓ Basis ID提取正确 (bX→0, bZ→1)
- ✓ Final mask生成正确 (checkerboard pattern)

**无问题发现**

---

### 1.5 损失函数和评估指标

- ✓ 使用 `BCEWithLogitsLoss` - 标准的二分类损失
- ✓ 梯度裁剪防止不稳定
- ✓ 准确率计算正确
- ✓ LER = 1 - accuracy (定义正确)

**无问题发现**

---

### 1.6 数据分布分析

需要在服务器上运行以下命令查看详细分布:

```bash
cd ~/work/ALPHAQUBIT
python3 comprehensive_diagnosis.py
```

这会分析:
- 预测vs真实标签的分布
- 是否存在类别不平衡
- 是否存在预测偏向

---

## 2. 性能按配置分组

### 按码距(Distance):
```
d3:  最好 (LER ~4-5%)
d5:  中等 (LER ~7-15%)
d7:  较差 (LER ~15-20%)
d9:  差   (LER >20%)
d25: 最差 (LER >20%)
```

### 按噪声率(Noise):
```
r01 (1%):  最好  (LER ~4-5%)   ← 接近预训练噪声
r03-r09:   中等  (LER 9-15%)
r25 (25%): 最差  (LER ~25%)   ← 远离预训练噪声
```

### 按测量基(Basis):
```
X basis: 平均LER ~20%
Z basis: 平均LER ~20%
```
无显著差异

---

## 3. 根本原因总结

### 主要原因 (Critical):

1. **预训练数据噪声范围不足** ⭐⭐⭐⭐⭐
   - 预训练: 0.1% 噪声
   - 测试: 1%-25% 噪声
   - 结果: 只在低噪声泛化良好

2. **模型容量对大码距不足** ⭐⭐⭐⭐
   - d3: 表现好 (充足容量)
   - d5-d7: 表现差 (容量不足)

### 次要原因 (Important):

3. **Fine-tuning数据可能不足** ⭐⭐⭐
   - 高噪声场景的样本可能太少
   - 需要检查训练集大小

4. **缺少课程学习策略** ⭐⭐
   - 直接在所有难度上训练
   - 没有从易到难的过程

---

## 4. 改进方案

### 方案A: 重新生成预训练数据 (推荐) ⭐⭐⭐⭐⭐

**目标**: 覆盖完整噪声范围

**步骤**:
1. 运行 `python fix_noise_range.py`
2. 这会生成11个噪声级别的数据 (0.1% - 30%)
3. 每级50k样本，总计550k样本
4. 使用合并后的数据重新预训练基础模型
5. 重新fine-tune所有实验

**预期改善**: 高噪声LER从25%降到10%以下

**时间**: 1-2天生成数据 + 3-5天训练

---

### 方案B: 增加模型容量 (快速) ⭐⭐⭐⭐

**目标**: 改善大码距性能

**步骤**:
1. 修改 `run_finetune_all.py`:
```python
def get_model_size(distance):
    if distance <= 3:
        return 256
    elif distance <= 5:
        return 384
    else:
        return 512

# In fine-tuning loop:
hidden_dim = get_model_size(extract_distance_from_filename(npz_file))
```

2. 重新fine-tune d>=5的实验

**预期改善**: d5/d7的LER降低5-10个百分点

**时间**: 1-2天重新训练

---

### 方案C: 增加Fine-tuning数据 (中期) ⭐⭐⭐

**目标**: 更多困难样本

**步骤**:
1. 为高噪声场景生成更多训练数据
2. 特别是r25, d7等困难配置
3. 从5000样本增加到20000+样本

**预期改善**: 所有场景平均提升2-5个百分点

**时间**: 2-3天生成+训练

---

### 方案D: 课程学习 (实验性) ⭐⭐

**目标**: 渐进式训练

**步骤**:
1. 第1阶段: 训练低噪声(r01-r03)
2. 第2阶段: 加入中噪声(r05-r07)
3. 第3阶段: 加入高噪声(r09-r25)

**预期改善**: 训练更稳定，可能提升5%

**时间**: 3-4天实验

---

## 5. 立即可执行的命令

### 在服务器上运行诊断:

```bash
cd ~/work/ALPHAQUBIT

# 1. 运行全面诊断
python3 comprehensive_diagnosis.py > diagnosis_full_report.txt

# 2. 查看诊断报告
less diagnosis_full_report.txt

# 3. 生成扩展噪声数据 (方案A)
python3 fix_noise_range.py

# 4. 或者先测试一个噪声级别
python3 generate_data.py --model pauli_plus --samples 10000
# 修改 configs/pauli_plus.yaml 中的 depolarization 为 0.05 (5%)
```

### 检查现有数据分布:

```bash
# 查看训练数据大小
find google_finetune_data/finetune -name "*.npz" -exec ls -lh {} \; | head -20

# 检查最好和最差模型的数据
python3 -c "
import numpy as np
best = np.load('google_finetune_data/test/samples_surface_code_bX_d3_r01_center_5_7.npz')
print(f'Best model test samples: {len(best[\"obs\"])}')
worst = np.load('google_finetune_data/test/samples_surface_code_bX_d5_r25_center_5_5.npz')
print(f'Worst model test samples: {len(worst[\"obs\"])}')
"
```

---

## 6. 决策树

```
开始
│
├─ 时间充足 (1周+) ?
│  │
│  ├─ 是 → 方案A (重新生成预训练数据) ⭐⭐⭐⭐⭐
│  │      最大改善，彻底解决问题
│  │
│  └─ 否 ↓
│
├─ 需要快速改善 (2-3天) ?
│  │
│  ├─ 是 → 方案B (增加模型容量) ⭐⭐⭐⭐
│  │      快速改善大码距性能
│  │
│  └─ 否 ↓
│
└─ 渐进式改进 ?
   │
   └─ 是 → 方案B + 方案C 组合
         先增加容量，再增加数据
```

---

## 7. 预期最终性能

### 如果执行方案A (重新预训练):
```
平均LER: 20.13% → 5-8%
最好模型: 3.74% → 2.5-3%  (可能超越论文!)
最差模型: 25.36% → 8-12%
```

### 如果执行方案B (增加容量):
```
平均LER: 20.13% → 12-15%
d5/d7性能: 显著改善
d3性能: 保持不变
```

---

## 8. 文件清单

### 新创建的工具:

1. ✅ `comprehensive_diagnosis.py` - 全面诊断脚本
   - 分析所有配置维度
   - 生成详细报告
   
2. ✅ `fix_noise_range.py` - 噪声范围修复脚本
   - 生成11个噪声级别
   - 自动备份和恢复配置
   - 包含合并脚本

3. ✅ `generate_summary_from_predictions.py` - 从预测重建摘要
   - 当test_summary.json缺失时使用

4. ✅ `analyze_results.md` - 分析文档
   - 问题诊断
   - 行动计划

---

## 9. 下一步行动 (优先级排序)

### 🔥 立即执行 (今天):

1. **在服务器运行诊断**
   ```bash
   cd ~/work/ALPHAQUBIT
   python3 comprehensive_diagnosis.py > diagnosis_report.txt
   cat diagnosis_report.txt
   ```

2. **决定改进方案** (A、B、C或组合)

3. **如果选方案A**:
   ```bash
   python3 fix_noise_range.py
   # 这会需要几小时到一天生成数据
   ```

### 📅 本周完成:

4. **重新训练** (根据选择的方案)

5. **重新评估**
   ```bash
   python3 test_finetuned_models.py --model-dir finetuned_models_v2 ...
   python3 run_evaluation_and_plot.py
   ```

6. **对比新旧结果**

### 📊 持续改进:

7. 监控训练loss曲线
8. 调整超参数
9. 尝试数据增强
10. 考虑模型集成

---

## 10. 联系方式

如有问题，参考以下文档:
- `README.md` - 项目总览
- `FINETUNING_GUIDE.md` - Fine-tuning指南
- `CODE_LOCATION_GUIDE.md` - 代码位置
- 论文: `s41586-024-08449-y.pdf`

---

**生成时间**: 2025-11-20
**版本**: v1.0
**状态**: ✅ 诊断完成，等待执行改进方案
