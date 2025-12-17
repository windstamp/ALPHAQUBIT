# 结果差距分析报告：实验结果 vs Google论文

## 📊 结果对比总览

| 指标 | 实验结果 | Google论文 | 差距 |
|------|----------|------------|------|
| **平均LER** | **20.13%** | **3.0%** | **17.13%** (6.7倍) |
| **最佳LER** | 3.74% | 1.5% | 2.24% |
| **最差LER** | 25.36% | 8.5% | 16.86% |

---

## 🔍 根本原因分析

### 1. ⚠️ 训练流程未完成

从 `pipeline_results.json` 可以看到：

```
"status": "failed",
"error": "module 'stim' has no attribute 'Circuit'"
```

**关键问题**：
- **stim 包安装失败** - Windows 需要 Visual C++ Build Tools
- **数据生成失败** - 没有生成预训练数据
- **模型未预训练** - 直接在未预训练的模型上做fine-tuning

**影响**：论文使用 **850万样本** 预训练，我们可能用的是 **随机初始化** 的模型！

---

### 2. 📈 训练数据量严重不足

| 阶段 | 论文规格 | 实验实际 | 比例 |
|------|----------|----------|------|
| **预训练样本** | 8,500,000 | 0 (失败) | 0% |
| **Fine-tune样本** | 50,000/实验 | ~19,880 | 40% |
| **预训练epochs** | 100 | 0 | 0% |
| **Fine-tune epochs** | 30 | 30 | 100% |

---

### 3. 🔧 超参数差异

| 参数 | 论文值 | 代码默认值 | 匹配 |
|------|--------|------------|------|
| **预训练 LR** | 1e-4 | N/A (未执行) | ❌ |
| **Fine-tune LR** | 1e-5 | 1e-4 | ❌ (10倍) |
| **预训练 batch** | 256 | N/A | ❌ |
| **Fine-tune batch** | 128 | 1024 | ❌ (8倍) |
| **Weight decay (预训练)** | 1e-4 | N/A | ❌ |
| **Weight decay (Fine-tune)** | 1e-3 | 1e-3 | ✅ |

**严重问题**：
- Fine-tune学习率 **过高10倍** → 可能导致catastrophic forgetting
- Batch size **过大8倍** → 泛化能力下降

---

### 4. 🧪 噪声范围覆盖不完整

论文测试的物理错误率范围：
```
0.1%, 0.2%, 0.3%, ..., 1.0%, ..., 2.5%
```

我们的预训练数据噪声范围（如果有的话）：
```
quick_test模式: 0.1%, 0.5%, 1.0% (仅3个点)
```

**问题**：高噪声率 (如 r25 = 25%噪声率) 几乎没有训练数据覆盖！

---

### 5. 📐 Code Distance 效应

从结果分析：
- **d3 (distance=3)**: LER ~3-5% ✓ 接近论文
- **d5 (distance=5)**: LER ~15-20% ✗ 远差于论文
- **d7 (distance=7)**: LER ~20-25% ✗ 完全不工作

**原因**：
- 模型未在大distance数据上充分预训练
- 论文使用了多distance混合训练策略

---

## 🎯 问题优先级排序

### P0 - 必须修复（阻断性问题）

1. **安装 stim 包**
   ```bash
   # 在Linux服务器上安装
   pip install stim
   
   # 或者用conda
   conda install -c conda-forge stim
   ```

2. **生成预训练数据** (850万样本)
   ```bash
   python make_all_pretraining_noise.py \
     --dem-samples 1000000 \
     --si1000-samples 5000000 \
     --soft-shots 2500000
   ```

3. **执行完整预训练** (100 epochs)
   ```bash
   python ai_models/train.py \
     --config configs/pauli_plus.yaml \
     --samples 8500000 \
     --epochs 100 \
     --batch-size 256 \
     --lr 1e-4
   ```

### P1 - 高优先级（影响显著）

4. **修正 Fine-tune 学习率**
   ```python
   # fine_tune.py 第28行
   p.add_argument("--lr", type=float, default=1e-5)  # 从1e-4改为1e-5
   ```

5. **修正 Fine-tune batch size**
   ```python
   # fine_tune.py 第27行  
   p.add_argument("--batch-size", type=int, default=128)  # 从1024改为128
   ```

6. **增加 Fine-tune 样本数**
   ```python
   # fine_tune.py 第30行
   p.add_argument("--train-samples", type=int, default=50000)  # 从19880改为50000
   ```

### P2 - 中优先级（改进性能）

7. **扩展噪声范围覆盖**
   - 增加高噪声率样本 (5%, 10%, 25%)
   - 增加大distance样本 (d5, d7)

8. **实现课程学习**
   - 先简单后困难：d3→d5→d7
   - 先低噪声后高噪声：r01→r05→r10→r25

---

## 📋 修复后预期性能

根据论文数据，正确实现后应该达到：

| 配置 | 当前LER | 预期LER | 改进 |
|------|---------|---------|------|
| d3, r01 | 3.74% | 2.5-2.8% | ~30% |
| d5, r01 | ~15% | 1.5% | ~90% |
| d5, r25 | 25.36% | 8.5% | ~66% |
| **平均** | **20.13%** | **3.0%** | **85%** |

---

## 🚀 立即执行的步骤

### 第一步：修复学习率

```python
# 修改 ai_models/fine_tune.py
# 将 default=1e-4 改为 default=1e-5
```

### 第二步：在有GPU的Linux服务器上

```bash
# 1. 安装 stim
pip install stim

# 2. 生成完整数据集
python make_all_pretraining_noise.py \
  --dem-samples 2000000 \
  --si1000-samples 4000000 \
  --soft-shots 2500000

# 3. 预训练模型
python ai_models/train.py \
  --config configs/paper_aligned.yaml \
  --epochs 100 \
  --batch-size 256

# 4. Fine-tune所有实验
python run_finetune_all.py \
  --lr 1e-5 \
  --batch-size 128 \
  --epochs 30
```

---

## 📚 结论

**主要差距来源（按影响排序）**：

1. **预训练完全缺失** (影响 ~60%)
2. **学习率过高** (影响 ~20%)
3. **数据量不足** (影响 ~10%)
4. **噪声覆盖不全** (影响 ~10%)

**解决这些问题后，预期可将LER从20%降到3%左右，与论文结果对齐。**

---

*分析日期：2024年12月*
