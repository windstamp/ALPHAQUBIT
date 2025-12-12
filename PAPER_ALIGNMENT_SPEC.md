# AlphaQubit 论文完整复现规格说明

## 论文信息

**标题**: Accurate neural network decoding of surface codes for quantum error correction  
**期刊**: Nature, 2024  
**DOI**: 10.1038/s41586-024-08449-y

---

## 1. 噪声模型规格 (Noise Model Specification)

### 1.1 Pauli+ 噪声模型 (Table S4)

论文使用的 Pauli+ 噪声模型包含以下物理机制：

| 参数 | 值 | 物理含义 | 代码位置 |
|------|-----|---------|---------|
| `cycle_ns` | 1076 ns | 表面码周期时间 | `my_noise_model/paper_aligned.py` |
| `T1_us` | 73 µs | 振幅阻尼时间 | `my_noise_model/paper_aligned.py` |
| `Tphi_us` | 720 µs | 纯退相干时间 | `my_noise_model/paper_aligned.py` |
| `p_readout` | 8.0×10⁻³ | 测量误差概率 | `my_noise_model/paper_aligned.py` |
| `p_reset` | 1.5×10⁻³ | 复位误差概率 | `my_noise_model/paper_aligned.py` |
| `p_heat_12` | 2.5×10⁻⁴ | \|1⟩→\|2⟩ 加热概率 | `my_noise_model/paper_aligned.py` |
| `p_cz_leak_11_to_02` | 2.0×10⁻⁴ | CZ诱导\|11⟩→\|02⟩泄漏 | `my_noise_model/paper_aligned.py` |
| `p_cz_crosstalk_ZZ` | 5.5×10⁻⁴ | ZZ串扰误差 | `my_noise_model/paper_aligned.py` |
| `p_1q_excess` | 6.2×10⁻⁴ | 单量子比特剩余Pauli误差 | `my_noise_model/paper_aligned.py` |
| `p_cz_excess` | 2.75×10⁻³ | CZ剩余Pauli误差 | `my_noise_model/paper_aligned.py` |

### 1.2 噪声通道实现

每个噪声通道都用 Kraus 算符实现，满足 CPTP 条件：

1. **振幅阻尼** (T₁衰减):
   - γ = 1 - exp(-t/T₁)
   - K₀ = [[1,0],[0,√(1-γ)]], K₁ = [[0,√γ],[0,0]]

2. **纯退相干** (T_φ):
   - p = (1 - exp(-t/T_φ)) / 2
   - K₀ = √(1-p)·I, K₁ = √p·Z

3. **GPTA (广义Pauli扭曲近似)**:
   - PTM对角元: λ[P] = (1/2ⁿ)·Tr(P†·E(P))
   - Hadamard变换: p = H·λ / 2ⁿ

4. **泄漏注入** (\|1⟩→\|2⟩):
   - 三能级系统，概率 p_heat_12

5. **DQLR复位矩阵**:
   ```
   |0⟩  |1⟩  |2⟩
   1.0  0.0  0.05  → |0⟩
   0.0  1.0  0.90  → |1⟩
   0.0  0.0  0.05  → |2⟩
   ```

6. **Soft XOR**:
   - soft_xor(p, q) = p + q - 2pq

7. **I/Q 读出模型**:
   - 三态: |0⟩, |1⟩, |L⟩ (泄漏)
   - 高斯后验概率分布

---

## 2. 数据规格 (Data Specification)

### 2.1 预训练数据

| 参数 | 值 | 说明 |
|------|-----|------|
| **总样本数** | 8,500,000 | 8.5M 模拟样本 |
| **噪声模型** | SI1000 + Pauli+ | 混合噪声 |
| **Code Distance** | d=3, d=5, d=7 | 三种码距 |
| **Rounds** | 变化 (1-25) | 不同轮数 |
| **Basis** | X 和 Z | 两种基 |

#### 2.1.1 按 Code Distance 分布
- d=3: ~2.8M 样本
- d=5: ~2.8M 样本  
- d=7: ~2.9M 样本

#### 2.1.2 按物理错误率分布 (SI1000)
- p ∈ {0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01}

### 2.2 微调数据

| 参数 | 值 | 说明 |
|------|-----|------|
| **每实验样本数** | 50,000 | Google QEC设备数据 |
| **训练/验证分割** | 80%/20% | 40K训练, 10K验证 |
| **Code Distance** | d=3, d=5 | 实际设备配置 |
| **Rounds** | r=1, r=5, r=10, r=25 | 不同轮数 |

### 2.3 测试数据

| 参数 | 值 | 说明 |
|------|-----|------|
| **每配置样本数** | 10,000 | 独立测试集 |
| **来源** | 保留的设备数据 | 不参与训练 |

---

## 3. 模型架构规格 (Model Architecture)

### 3.1 Transformer 配置

| 模型大小 | Hidden Dim | Heads | Layers | 参数量 |
|---------|-----------|-------|--------|--------|
| Small | 64 | 4 | 4 | ~0.5M |
| Medium | 128 | 8 | 8 | ~2M |
| **Large** | **256** | **8** | **12** | **~8M** |
| XLarge | 512 | 16 | 16 | ~32M |

**默认使用 Large 模型**

### 3.2 位置编码

- `row_emb`: 行位置嵌入
- `col_emb`: 列位置嵌入
- `dx_emb`: 行位移嵌入
- `dy_emb`: 列位移嵌入
- `manh_emb`: 曼哈顿距离嵌入
- `same_emb`: 同奇偶性指示嵌入

### 3.3 注意力机制

- **缩放因子**: 1/√d_k = 1/√32 ≈ 0.1768
- **Pre-LayerNorm**: 先归一化再计算
- **FFN扩展**: 4× hidden_dim
- **激活函数**: SiLU/Swish with gating

---

## 4. 训练超参数 (Training Hyperparameters)

### 4.1 预训练

| 参数 | 值 |
|------|-----|
| **样本数** | 8,500,000 |
| **Batch Size** | 256 |
| **Learning Rate** | 1×10⁻⁴ |
| **Epochs** | 100 |
| **Optimizer** | AdamW |
| **Weight Decay** | 1×10⁻⁴ |
| **LR Scheduler** | Cosine Annealing |
| **Loss** | BCEWithLogitsLoss |

### 4.2 微调

| 参数 | 值 |
|------|-----|
| **样本数** | 50,000 (每实验) |
| **Batch Size** | 128 |
| **Learning Rate** | 1×10⁻⁵ |
| **Epochs** | 30 |
| **Optimizer** | AdamW |
| **Weight Decay** | 1×10⁻³ |
| **Early Stopping** | patience=5 |
| **Gradient Clipping** | max_norm=1.0 |

---

## 5. 评估指标 (Evaluation Metrics)

### 5.1 主要指标

- **Logical Error Rate (LER)**: 逻辑错误率
  ```python
  LER = (predictions != labels).mean()
  ```

- **Accuracy**: 准确率
  ```python
  Accuracy = (predictions == labels).mean() = 1 - LER
  ```

### 5.2 阈值 (Threshold)

- **AlphaQubit**: p_th ≈ 0.82%
- **MWPM**: p_th ≈ 0.69%

---

## 6. 论文图表复现 (Figure Reproduction)

### Figure 2: Threshold Plot
- X轴: 物理错误率 (0.001 - 0.01)
- Y轴: 逻辑错误率 (log scale)
- 曲线: d=3, d=5, d=7 for AlphaQubit vs MWPM

### Figure 3: Decoder Comparison
- 比较: AlphaQubit, MWPM, Tensor Network, BP, Union Find
- 条件: SI1000 p=0.005/0.01, d=5

### Figure 4: Fine-tuning Results  
- Google QEC v3.5 实验数据
- 多个实验配置的LER比较

### Extended Data: Ablations
- 模型大小消融
- 预训练数据量消融
- Soft vs Hard 读出消融

---

## 7. 文件结构规范

### 7.1 数据目录结构
```
pretrain_data/
├── dem/
│   ├── dem_syndromes_d3_r25.npy
│   ├── dem_logicals_d3_r25.npy
│   └── ...
├── si1000/
│   ├── si1000_syndromes_p0.001_d3.npy
│   ├── si1000_logicals_p0.001_d3.npy
│   └── ...
└── pauli_plus/
    ├── samples_surface_code_bX_d3_r01.npz
    └── ...

finetune_data/
├── train/
│   └── samples_*.npz
└── val/
    └── samples_*.npz

test_data/
└── samples_*.npz
```

### 7.2 结果目录结构
```
results/
├── YYYYMMDD_HHMMSS/
│   ├── config.json           # 运行配置
│   ├── pretrain/
│   │   ├── model_epoch_*.pth
│   │   └── train_log.json
│   ├── finetune/
│   │   ├── exp_*/
│   │   │   ├── model_best.pth
│   │   │   └── metrics.json
│   │   └── summary.json
│   ├── test/
│   │   ├── predictions_*.npz
│   │   └── metrics.json
│   └── figures/
│       ├── fig2_threshold.png
│       ├── fig3_comparison.png
│       └── fig4_finetuning.png
```

---

## 8. 代码对应关系

| 论文章节 | 代码文件 |
|---------|---------|
| Methods: Model Architecture | `ai_models/model.py`, `ai_models/model_mla.py` |
| Methods: Noise Model | `my_noise_model/paper_aligned.py`, `my_noise_model/channels.py` |
| Methods: GPTA | `my_noise_model/gpta.py` |
| Methods: Soft Readout | `my_noise_model/iq_readout.py`, `my_noise_model/softxor.py` |
| Methods: Training | `ai_models/train.py` |
| Supplementary Table S4 | `my_noise_model/paper_aligned.py` |
| Figure 2 | `paper_figures/fig2_threshold_plot.py` |
| Figure 3 | `paper_figures/fig3_decoder_comparison.py` |
| Figure 4 | `paper_figures/fig4_finetuning_results.py` |

---

## 9. 验证检查清单

- [ ] Table S4 所有10个参数完全匹配
- [ ] 所有7个Kraus通道满足CPTP条件
- [ ] 模型架构: 12层, 8头, 256维
- [ ] 预训练: 8.5M样本, batch=256, lr=1e-4
- [ ] 微调: 50K样本, batch=128, lr=1e-5
- [ ] Soft XOR公式正确: p + q - 2pq
- [ ] GPTA实现正确
- [ ] DQLR矩阵为有效随机矩阵
- [ ] 所有图表可复现

---

*文档版本: 1.0*  
*最后更新: 2024年12月10日*
