# AlphaQubit：量子纠错模拟与机器学习解码工具包

## 概述

AlphaQubit 提供从量子纠错仿真数据生成、模型训练到解码评估的完整流程。仓库包含可配置的噪声模型、基于 PyTorch 的训练脚本以及数据可视化工具，帮助研究者快速开展量子误码校正实验。

## 功能特性

- **数据生成**：支持检测误差模型（DEM）、电路去极化噪声（SI1000）以及泄漏、串扰与软读出（Pauli+）等多种噪声类型。
- **配置实验**：`configs/` 目录提供 YAML 配置文件，可灵活设置噪声与训练参数。
- **表面码仿真**：`simulator/` 内含表面码仿真器，实现论文级别的物理噪声模型。
- **模型管理**：训练脚本统一把权重保存至 `ai_models/models/`，解码脚本会在该目录及仓库根目录、`ai_models/checkpoints/` 等位置自动搜寻 `.pth` 文件。
- **可视化工具**：提供 `npy_viewer.py`（.npy 查看）与 `plot_alphaqubit_results.py`（性能曲线绘制）。

## 模型架构与数据说明

- **基础设施**：解码器基于 PyTorch 实现，核心由三部分组成：稳定子特征嵌入器、堆叠的改良 Transformer 层（内部使用 DeepSeek MLA 多头注意力）、以及读取逻辑误差对数似然的卷积+MLP 头部。【F:ai_models/model_mla.py†L16-L143】
- **输入特征**：每个样本的张量形状为 `(N, R, S, 3)`，其中第 0 通道是离散检测事件比特，第 1、2 通道分别为软读出后计算态与泄漏态的后验概率；这些通道由仿真脚本在采样时将检测比特与两路 soft channel 拼接而成。【F:google_qec_simulator/data_manager.py†L5-L28】【F:google_qec_simulator/experiment_simulator.py†L24-L33】
- **输出含义**：模型输出单个标量对数几率（logit），与 `BCEWithLogitsLoss` 搭配训练，表示对应逻辑比特发生错误的概率；推理时可通过 `torch.sigmoid` 转换为 [0,1] 概率。【F:ai_models/model_mla.py†L79-L143】【F:ai_models/model_mla.py†L188-L194】
- **参数规模**：针对随仓库提供的 `d=5` 软读出样例（稳定子数 24、特征通道 3），默认隐藏维度 256、12 层、8 头的 AlphaQubit 解码器共有 8,242,177 个可训练参数。【354d2a†L1-L27】
- **训练样本**：示例数据集 `simulated_data/samples_surface_code_bX_d5_r01_center_5_5.npz` 含 20,000 条样本，形状 `(20000, 1, 24, 3)`；批量预训练脚本 `make_all_pretraining_noise.py` 的默认配置会合成约 $8.5\times10^6$ 条离散综合样本与 4.0M 次 soft shots。【96f910†L1-L8】【F:README.md†L102-L103】

## 安装

```bash
git clone https://github.com/xuda1979/ALPHAQUBIT.git
cd ALPHAQUBIT
# 可选：创建并激活虚拟环境
python3.8 -m venv venv
source venv/bin/activate
# 安装依赖（任选其一）
pip install --upgrade pip
pip install numpy scipy stim pyyaml torch leakysim>=0.4.0
# 或使用项目罗列的依赖清单
pip install -r requirements.txt

# 如需在华为 Ascend NPU 上训练，请安装带有 torch.npu 的 PyTorch 发行版并根据官方文档完成驱动配置。
```

## 仓库结构

```plaintext
├── ai_models/                   # 模型训练与解码脚本
├── configs/                     # 实验配置 YAML 文件
├── simulator/                   # 量子纠错仿真器
├── simulated_data/              # google_qec_simulator 生成的 .npz（脚本可自动建立）
├── paper_figures/               # 论文图表生成脚本
│   ├── output/                  # 生成的图表与表格
│   ├── paper_data.py            # 论文参考数据
│   ├── fig2_threshold_plot.py   # 图 2：阈值曲线
│   ├── fig3_decoder_comparison.py # 图 3：解码器对比
│   ├── fig4_finetuning_results.py # 图 4：微调结果
│   ├── extended_ablations.py    # 扩展数据：消融实验
│   └── generate_all.py          # 一键生成所有图表
├── generate_data.py             # 训练数据生成脚本
├── npy_viewer.py                # .npy 数据查看工具
├── plot_alphaqubit_results.py   # 解码性能绘制脚本
├── models/                      # 可选：手动放置 .pth 权重的目录
└── README.md                    # 项目说明
```

> **提示**：真实实验 Stim 电路存放在外部目录
> ``~/work/google_qec3v5_experiment_data``，仓库内不再附带 `experiment_data/` 副本。

## 工作流程

### 1. 生成噪声与综合数据

使用 `generate_data.py` 可按噪声模型生成样本：

```bash
# 检测误差模型（DEM）
python generate_data.py --model dem --samples 10000

# 电路去极化噪声（SI1000）
python generate_data.py --model si1000 --samples 10000

# 泄漏、串扰与软读出（Pauli+）
python generate_data.py --model pauli_plus --samples 10000

# 与论文对齐的物理噪声模型（支持 X/Z 基）
python generate_data.py --model paper_aligned --basis z --samples 10000
python generate_data.py --model paper_aligned --basis x --samples 10000
```

- 默认输出目录为 `output/`，文件名含时间戳，如 `output/dem_syndromes_z_YYYYMMDD_HHMMSS.npy`。
- `google_qec_simulator` 可直接生成 `.npz` 噪声文件，并通过 `--device {cpu,cuda,npu}` 指定计算硬件：

```bash
python -m google_qec_simulator.main path/to/exp --shots 10000 --device npu
# 默认输出写入 pretrain_data/samples_<实验文件夹>.npz，可用 --out 自定义位置
```

#### 预生成大规模预训练数据

仓库提供一键脚本，用于生成论文同等规模的预训练噪声：

```bash
python make_all_pretraining_noise.py \
  --dem-samples 2500000 \
  --si1000-samples 1500000 \
  --si1000-p-grid 0.004,0.008,0.012,0.016 \
  --soft-shots 4000000 \
  --soft-device auto \
  --out-dir pretrain_data
# 若实验 Stim 位于额外目录（例如 Google QEC 数据集），可重复传入 --experiment-root：
# python make_all_pretraining_noise.py --out-dir pretrain_data --experiment-root ~/work/google_qec3v5_experiment_data
```

该脚本会依次调用 `generate_data.py`（DEM/SI1000）与 `run_create_all_samples.py`（soft/IQ），并把生成的 `.npy`/`.npz` 文件收集到 `--out-dir` 指定的目录，同时写出 `MANIFEST.json`（记录时间戳与所有产物）。soft/IQ 噪声会按照实验名称自动分目录存放，例如 `pretrain_data/<experiment>/samples_*.npz`，便于按实验拆分训练数据。上述配置对应约 $8.5\times10^6$ 条离散综合样本与 4.0M 次 soft shots，需要约 15 GB（布尔综合）+1.1 GB（soft shots）存储。如资源受限，可按比例缩放各 `--*-samples`，保持不同噪声类型的相对比重。

#### 批量生成实验数据

若 `~/work/google_qec3v5_experiment_data/` 中含有多个实验子目录，可批量调用模拟器生成 `samples_<experiment>.npz`：

```bash
python run_create_all_samples.py --shots 2000
python run_create_all_samples.py --skip-existing --device npu  # 支持跳过已生成文件与 NPU 加速
# 支持多目录：python run_create_all_samples.py ~/work/google_qec3v5_experiment_data /path/to/others
```

脚本会递归查找含 `.stim` 的实验目录，将输出写入 `pretrain_data/`，并按相对路径命名，例如 `pretrain_data/samples_folder_subfolder.npz`。
该流程直接调用 `google_qec_simulator.main.simulate_folder` 完成采样，因此运行期间会实时输出 `[scan]`/`[sample]` 等进度信息，便于观察当前实验与累计耗时。
因此上述单行命令会自动遍历 `~/work/google_qec3v5_experiment_data/` 中的全部实验并依次生成噪声。

### 2. 检查与浏览数据

列出 `output/` 下的 `.npy` 文件并调用 `npy_viewer.py`：

```bash
ls output/*.npy
python npy_viewer.py output/dem_syndromes_z_20240229_101530.npy
```

### 3. 训练模型

使用单个配置文件训练：

```bash
python ai_models/train.py --config configs/dem.yaml
```

- 可直接在 YAML 中调整超参数与噪声设置。
- `ai_models/train.py` 默认将 `dem/si1000/pauli_plus/paper_aligned` 对应模型保存为仓库根目录下的 `alphaqubit_<模型类型>.pth`，可通过 `--model-path` 修改输出位置。

批量训练 `pretrain_data/`（或其它目录）中的所有实验：

```bash
python run_training_all.py
python run_training_all.py --npu  # 自动检测 Ascend NPU 并行调度
python run_training_all.py --epochs 1 --batch-size 32 --max-samples 1024  # 本地快速冒烟
# 自动生成缺失数据时可指定 Stim 根目录：
# python run_training_all.py --experiment-root ~/work/google_qec3v5_experiment_data
```

在运行批量训练前，建议先执行上一节的一键脚本：

```bash
python make_all_pretraining_noise.py --out-dir pretrain_data
```

它会为每个实验生成并复制 soft/IQ `.npz` 至 `pretrain_data/<experiment>/`，随后 `run_training_all.py` 会优先遍历该目录为每个数据集调用 `ai_models/model_mla.py` 并把权重写入 `ai_models/models/NAME.pth`；若 `pretrain_data/` 为空，则自动回退到 `simulated_data/**/*.npz`。通过 `--epochs`、`--batch-size` 与 `--max-samples` 可以快速调整训练时长；其中 `--max-samples` 会在加载数据集后裁剪样本数，便于在 CPU 上进行冒烟测试。如需自定义数据目录，可重复传入 `--data-root <path>`，该选项会覆盖默认搜索路径（因此若要保留默认目录，请一并显式给出）。脚本会自动串行调度任务；在支持 Ascend NPU 并传入 `--npu` 时，会检测可用设备并并行分配训练进程，因此无需单独的串行脚本。

### 4. 解码与评估

#### 单个数据集

```bash
python ai_models/decode.py \
  --model finetuned_models/alphaqubit_sycamore_runs.pth \
  --data output/dem_syndromes_z_20240229_101530.npy
```

结果写入 `results/`。如需自定义目录，可使用 `--results-dir`。示例命令默认假设权重来自仓库根目录下的 `finetuned_models/` 文件夹——`ai_models/fine_tune.py` 与批量脚本 `run_fine_tune_all.py` 会在该目录生成面向具体实验微调过的 `alphaqubit_<folder>.pth` 权重，解码阶段推荐直接使用这些模型。

#### 批量解码

```bash
python run_decode_all.py --model finetuned_models/
python run_decode_all.py --model finetuned_models/alphaqubit_sycamore_runs.pth
```

- `--model` 支持传入单个文件、目录或留空（若仅检测到一个模型则自动使用）。
- 当仅提供文件名时，脚本会优先在仓库根目录下的 `finetuned_models/` 目录查找微调权重，随后再依次检查 `ai_models/checkpoints/`、`ai_models/models/`、`checkpoints/` 与 `models/` 等位置。
- 默认遍历 `output/`，可通过 `--data-root` 或通配符（如 `"simulated_data/*.npz"`）指定其它数据源。
- `--predictions-dir` 保存逐次测量概率，`--skip-existing` 跳过已生成指标，`--dry-run` 仅打印执行计划。
- 若未生成综合数据，脚本会提示未找到解码目标。请使用前述数据生成脚本准备独立采样的测试集，以可靠评估泛化性能。

批量脚本同样适用于微调后模型：`ai_models/fine_tune.py` 与 `run_fine_tune_all.py` 会在 `finetuned_models/` 目录生成 `alphaqubit_<folder>.pth`，也会被自动发现。

所有批量解码结果默认写入 `results/`，若传入目录则为每个模型创建子目录（例如 `results/alphaqubit_dem/`），同时可通过 `--predictions-dir` 输出 `*_probs.npy`。

### 5. 可视化

`plot_alphaquibit_results.py` 可以对经过微调的实验文件夹计算 LER 并绘制柱状图：

```bash
python plot_alphaquibit_results.py --data-root ~/work/google_qec3v5_experiment_data/sycamore_runs --model-dir finetuned_models
```

- `--data-root`：包含多个实验子目录（每个子目录需含 `detection_events.b8`、`obs_flips_actual.01` 等文件）。
- `--model-dir`：对应的 `alphaqubit_<folder>.pth` 模型所在目录，默认为 `finetuned_models/`。

脚本会读取各子目录、按文件名解析轮数 `rXX`，再载入同名权重计算逻辑错误率，并绘制汇总图。

## Paper-aligned 噪声模型参数

`configs/paper_aligned.yaml` 提供了与 Google 论文中物理机制一一对应的参数：

| 参数 | 物理机制 |
| --- | --- |
| `T1_us`, `Tphi_us` | 振幅/相位弛豫，经 GPT 处理后注入到所有 1Q 门 |
| `p_cz_crosstalk_ZZ` | 并行 CZ 的 ZZ 串扰，作为相关错误注入 |
| `p_cz_swap_like` | CZ 期间的 swap-like 误差，建模为 (XX+YY)/2 |
| `p_cz_leak_11_to_02` | 相位诱导的泄漏，近似为相关 ZZ 误差 |
| `p_leak_transport_12_to_30` | 泄漏传输，引入附加的单量子比特 Pauli 噪声 |
| `p_readout`, `p_reset` | 测量与复位的经典翻转误差 |

## 论文图表生成

`paper_figures/` 目录包含用于复现 AlphaQubit 论文所有图表的脚本。

**论文来源**：*"Accurate neural network decoding of surface codes for quantum error correction"* - Nature, 2024 ([DOI: 10.1038/s41586-024-08449-y](https://doi.org/10.1038/s41586-024-08449-y))

### 快速开始

一键生成所有图表：

```bash
python -m paper_figures.generate_all
```

交互式显示图表：

```bash
python -m paper_figures.generate_all --show
```

### 生成的文件

所有输出保存至 `paper_figures/output/`：

#### 图表文件

| 文件 | 说明 |
|------|------|
| `fig2_threshold_plot.png/pdf` | 阈值与缩放行为（主图） |
| `fig2_threshold_comparison.png` | AlphaQubit vs MWPM 阈值对比 |
| `fig3_decoder_comparison.png/pdf` | 解码器性能对比柱状图 |
| `fig3_improvement_chart.png` | AlphaQubit 相对其他解码器的提升 |
| `fig3_radar_chart.png` | 多维度解码器对比雷达图 |
| `fig4_finetuning_results.png/pdf` | Google QEC 微调结果 |
| `fig4_grouped_analysis.png` | 按码距和噪声分组的 LER |
| `fig4_heatmap.png` | 不同配置下的 LER 热力图 |
| `ext_ablation_*.png` | 扩展数据：消融实验（3 张图） |

#### 表格文件

| 文件 | 说明 |
|------|------|
| `table1_architecture.txt` | 模型架构配置 |
| `table2_training.txt` | 训练超参数 |
| `table3_decoder_comparison.txt` | 解码器性能对比 |
| `table4_finetuning.txt` | 微调结果汇总 |

### 单独生成特定图表

```bash
# 图 2：阈值曲线
python -m paper_figures.fig2_threshold_plot

# 图 3：解码器对比
python -m paper_figures.fig3_decoder_comparison

# 图 4：微调结果
python -m paper_figures.fig4_finetuning_results

# 扩展数据：消融实验
python -m paper_figures.extended_ablations

# 仅生成表格
python -m paper_figures.generate_tables_simple
```

### 参考数据

`paper_figures/paper_data.py` 包含论文中的所有参考数值：

- **阈值数据**：d=3,5,7 下逻辑错误率与物理错误率的关系
- **解码器对比**：AlphaQubit、MWPM、张量网络、BP、UF
- **微调结果**：Google QEC v3.5 实验结果
- **消融实验**：软读出、预训练策略、模型规模
- **模型架构**：从 small 到 xlarge 的配置
- **训练配置**：预训练与微调超参数

## 参考建议

- 建议使用独立采样的数据集进行评估，避免与训练数据重合。
- 在支持的硬件上使用 `--device npu` 或 `--device cuda` 可显著加速 soft 通道采样与模型训练。

---

## 详细技术规格（与论文完全对齐）

本节详细记录了 AlphaQubit 复现实现的所有技术细节，确保与原始论文 *"Accurate neural network decoding of surface codes for quantum error correction"* (Nature, 2024, DOI: 10.1038/s41586-024-08449-y) 完全一致。

### 1. 模型架构详解

#### 1.1 整体架构

AlphaQubit 解码器采用修改版 Transformer 架构，专门针对表面码综合征解码任务优化：

```
输入 → 稳定子嵌入器 → N层Transformer → 读出网络 → 逻辑错误概率
```

#### 1.2 架构参数（Large 模型，论文默认）

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_dim` | 256 | 隐藏层维度 |
| `num_heads` | 8 | 注意力头数 |
| `num_layers` | 12 | Transformer 层数 |
| `head_dim` | 32 | 每个注意力头的维度 (256/8) |
| `ffn_dim` | 1024 | 前馈网络维度 (4×hidden_dim) |
| `总参数量` | ~8M | 约 8,242,177 个可训练参数 |

#### 1.3 模型变体

| 变体 | hidden_dim | num_heads | num_layers | 参数量 |
|------|-----------|-----------|------------|--------|
| Small | 64 | 4 | 4 | ~0.5M |
| Medium | 128 | 8 | 8 | ~2M |
| **Large** | **256** | **8** | **12** | **~8M** |
| XLarge | 512 | 16 | 16 | ~32M |

#### 1.4 组件详解

**稳定子嵌入器 (StabilizerEmbedder)**
- 为每个特征通道创建独立的线性投影
- 添加位置嵌入（索引嵌入）
- 添加最终轮次标记（on/off 嵌入）
- 输出经过 LayerNorm 归一化

**Transformer 层 (SyndromeTransformerLayer)**
- 采用 Pre-LayerNorm 架构（先归一化再计算）
- 多头注意力使用 DeepSeek MLA（Multi-head Latent Attention）
- 注意力缩放因子：$\frac{1}{\sqrt{d_k}} = \frac{1}{\sqrt{32}} \approx 0.1768$
- 前馈网络使用 GELU 激活函数
- 可选：膨胀卷积用于捕获局部空间结构

**位置编码**
- `row_emb`: 行位置嵌入
- `col_emb`: 列位置嵌入
- `dx_emb`: 行位移嵌入
- `dy_emb`: 列位移嵌入
- `manh_emb`: 曼哈顿距离嵌入
- `same_emb`: 同奇偶性指示嵌入

**读出网络 (ReadoutNetwork)**
- 2×2 卷积层将稳定子网格转换为数据量子比特网格
- 根据测量基（X/Z）沿不同轴聚合
- MLP 输出最终 logit

#### 1.5 代码位置

```
ai_models/model.py          # 标准 Transformer 实现
ai_models/model_mla.py      # MLA 注意力变体
mla/core.py                 # DeepSeek MLA 核心实现
mla/attention.py            # 注意力机制
```

### 2. 输入数据格式

#### 2.1 数据形状

```python
输入张量形状: (N, R, S, F)
- N: 批次大小
- R: QEC 轮数
- S: 稳定子数量 (对于距离 d，约为 (d-1)² × R)
- F: 特征通道数
```

#### 2.2 特征通道

| 通道 | 内容 | 数值范围 |
|------|------|----------|
| 0 | 检测事件（硬读出） | {0, 1} |
| 1 | P(|1⟩) 软读出后验 | [0, 1] |
| 2 | P(泄漏) 软读出后验 | [0, 1] |
| 附加 | 基ID（X=0, Z=1） | {0, 1} |

#### 2.3 输出格式

```python
输出: 标量 logit
- 训练时与 BCEWithLogitsLoss 配合使用
- 推理时: prob = torch.sigmoid(logit)
- 阈值: prediction = 1 if prob > 0.5 else 0
```

### 3. 数据规格

#### 3.1 预训练数据

| 参数 | 值 | 说明 |
|------|-----|------|
| **总样本数** | **8,500,000** | 8.5M 模拟样本 |
| 噪声模型 | SI1000 + Pauli+ | 混合噪声 |
| 码距 | d=3, d=5, d=7 | 三种码距 |
| 轮数 | 1, 5, 10, 25 | 不同轮数 |
| 基 | X 和 Z | 两种测量基 |

**按码距分布**
- d=3: ~2.85M 样本
- d=5: ~2.85M 样本
- d=7: ~2.85M 样本

**SI1000 物理错误率网格**
```python
p_grid = [0.001, 0.002, 0.003, 0.004, 0.005, 
          0.006, 0.007, 0.008, 0.009, 0.010]
```

每个 (距离, 错误率) 配置约 285,000 个样本：
$$\frac{8,500,000}{3 \times 10} \approx 283,333$$

#### 3.2 微调数据

| 参数 | 值 | 说明 |
|------|-----|------|
| **每实验样本数** | **50,000** | Google QEC 设备数据 |
| 训练/验证分割 | 80%/20% | 40K训练, 10K验证 |
| 码距 | d=3, d=5 | 实际设备配置 |
| 轮数 | r=1, 5, 10, 25 | 不同轮数 |

#### 3.3 测试数据

| 参数 | 值 | 说明 |
|------|-----|------|
| **每配置样本数** | **10,000** | 独立测试集 |
| 来源 | 保留的设备数据 | 不参与训练 |

### 4. 训练超参数

#### 4.1 预训练配置

| 参数 | 值 | 代码位置 |
|------|-----|----------|
| 样本数 | 8,500,000 | `run_server_pipeline.py` |
| Batch Size | **256** | `PipelineConfig.pretrain_batch_size` |
| Learning Rate | **1×10⁻⁴** | `PipelineConfig.pretrain_lr` |
| Epochs | **100** | `PipelineConfig.pretrain_epochs` |
| Optimizer | **AdamW** | `torch.optim.AdamW` |
| Weight Decay | **1×10⁻⁴** | `PipelineConfig.pretrain_weight_decay` |
| LR Scheduler | **Cosine Annealing** | `CosineAnnealingLR` |
| Loss | BCEWithLogitsLoss | `nn.BCEWithLogitsLoss()` |

#### 4.2 微调配置

| 参数 | 值 | 代码位置 |
|------|-----|----------|
| 样本数 | 50,000/实验 | `PipelineConfig.finetune_samples_per_exp` |
| Batch Size | **128** | `PipelineConfig.finetune_batch_size` |
| Learning Rate | **1×10⁻⁵** | `PipelineConfig.finetune_lr` |
| Epochs | **30** | `PipelineConfig.finetune_epochs` |
| Optimizer | AdamW | 同上 |
| Weight Decay | **1×10⁻³** | `PipelineConfig.finetune_weight_decay` |
| Early Stopping | **patience=5** | `PipelineConfig.finetune_patience` |
| Gradient Clipping | **max_norm=1.0** | `clip_grad_norm_` |
| LR Scheduler | OneCycleLR | 单周期学习率 |

### 5. 噪声模型详解

#### 5.1 Table S4 参数（Pauli+ 噪声模型）

以下参数来自论文补充材料 Table S4，在 `my_noise_model/paper_aligned.py` 中实现：

| 参数 | 值 | 单位 | 物理含义 |
|------|-----|------|---------|
| `cycle_ns` | 1076.0 | ns | 表面码周期时间 |
| `T1_us` | 73.0 | μs | 振幅阻尼时间常数 |
| `Tphi_us` | 720.0 | μs | 纯退相干时间常数 |
| `p_readout` | 8.0×10⁻³ | - | 测量误差概率 |
| `p_reset` | 1.5×10⁻³ | - | 复位误差概率 |
| `p_heat_12` | 2.5×10⁻⁴ | - | \|1⟩→\|2⟩ 加热概率 |
| `p_cz_leak_11_to_02` | 2.0×10⁻⁴ | - | CZ诱导 \|11⟩→\|02⟩ 泄漏 |
| `p_cz_crosstalk_ZZ` | 5.5×10⁻⁴ | - | ZZ 串扰误差 |
| `p_1q_excess` | 6.2×10⁻⁴ | - | 单量子比特剩余 Pauli 误差 |
| `p_cz_excess` | 2.75×10⁻³ | - | CZ 剩余 Pauli 误差 |

#### 5.2 DQLR 复位矩阵

数据量子比特泄漏复位 (DQLR) 不完美性由以下转移矩阵描述：

```
        |0⟩    |1⟩    |2⟩   (初始态)
|0⟩  [ 1.00   0.00   0.05 ]
|1⟩  [ 0.00   1.00   0.90 ]  (末态)
|2⟩  [ 0.00   0.00   0.05 ]
```

含义：
- |0⟩, |1⟩ 保持不变
- |2⟩（泄漏态）有 90% 概率复位到 |1⟩，5% 到 |0⟩，5% 保持泄漏

#### 5.3 Kraus 通道实现

所有噪声通道均通过 Kraus 算符实现，满足 CPTP（完全正迹保持）条件：

**振幅阻尼 (T₁ 衰减)**
```python
γ = 1 - exp(-t/T₁)
K₀ = [[1, 0], [0, √(1-γ)]]
K₁ = [[0, √γ], [0, 0]]
```

**纯退相干 (T_φ)**
```python
p = (1 - exp(-t/T_φ)) / 2
K₀ = √(1-p) · I
K₁ = √p · Z
```

**泄漏注入 (|1⟩→|2⟩)**
- 三能级系统，以概率 `p_heat_12` 将 |1⟩ 泵浦到泄漏态 |2⟩

**代码位置**: `my_noise_model/channels.py`

#### 5.4 广义 Pauli 托恩近似 (GPTA)

GPTA 将任意噪声通道转换为等效 Pauli 通道，便于 Clifford 仿真：

1. **计算 PTM 对角元**:
   $$\lambda[P] = \frac{1}{2^n} \text{Tr}(P^\dagger \cdot \mathcal{E}(P))$$

2. **Hadamard 变换得概率**:
   $$\mathbf{p} = \frac{H \cdot \boldsymbol{\lambda}}{2^n}$$

**代码位置**: `my_noise_model/gpta.py`

#### 5.5 软读出模型 (I/Q Readout)

软测量使用一维高斯分布模拟 I/Q 读出：

| 状态 | 均值 | 标准差 |
|------|------|--------|
| \|0⟩ | +SNR/2 | σ |
| \|1⟩ | -α·SNR/2 (α=exp(-τ)) | σ |
| \|L⟩ (泄漏) | 0 | 1.6σ |

**后验概率计算**:
```python
P(state|x) = prior(state) × Gaussian(x|μ_state, σ_state) / Z
```

**代码位置**: `my_noise_model/iq_readout.py`

#### 5.6 Soft XOR 公式

用于组合连续检测事件的软概率：

$$\text{soft\_xor}(p, q) = p + q - 2pq$$

**代码位置**: `my_noise_model/softxor.py`

### 6. 评估指标

#### 6.1 逻辑错误率 (LER)

```python
LER = (predictions != labels).mean()
# 或等价地
LER = 1 - Accuracy
```

对于多轮采样：
```python
LER = samples.any(axis=1).mean()  # 任意轮有错误即计为逻辑错误
```

#### 6.2 阈值

| 解码器 | SI1000 阈值 |
|--------|-------------|
| AlphaQubit | p_th ≈ **0.82%** |
| MWPM | p_th ≈ 0.69% |

### 7. 运行完整流水线

#### 7.1 服务器端一键运行

```bash
# 切换到项目目录
cd /home/ma-user/work/ALPHAQUBIT

# 快速测试（约 10-30 分钟）
python run_complete_server_pipeline.py --quick-test --output-dir quick_results

# 完整论文复现（约 24-48 小时）
python run_complete_server_pipeline.py --output-dir full_results --device auto
```

#### 7.2 流水线阶段

1. **论文对齐验证** - 检查所有参数是否与论文一致
2. **数据生成** - 生成 SI1000、Pauli+、测试数据
3. **预训练** - 8.5M 样本，100 epochs
4. **微调** - 每个实验 50K 样本，30 epochs
5. **测试** - 10K 样本评估
6. **报告生成** - Markdown 研究报告

#### 7.3 运行命令示例

```bash
# 快速测试（小数据，约 10-30 分钟）
python run_complete_server_pipeline.py --quick-test --output-dir test_results

# 完整论文复现（约 24-48 小时）
python run_complete_server_pipeline.py --output-dir full_results --device auto

# 仅运行论文对齐检查
python run_paper_alignment_check.py --output-dir alignment_results
```

#### 7.4 输出文件位置详解

**完整流水线输出目录结构**:

```
<output-dir>/                       # 例如: test_results/ 或 full_results/
├── 01_alignment/                   # Stage 1: 论文对齐验证
│   ├── alignment_report.json       # 对齐检查结果 (机器可读)
│   └── alignment_report.md         # 对齐检查报告 (人类可读)
│
├── 02_data/                        # Stage 2: 生成的训练/测试数据
│   ├── si1000/                     # SI1000 预训练数据
│   │   └── si1000_d{d}_p{p}.npz   # 按距离和错误率组织
│   ├── pauli_plus/                 # Pauli+ 微调数据
│   │   └── samples_surface_code_b{X/Z}_d{d}_r{r}.npz
│   └── test/                       # 测试数据
│       └── test_b{X/Z}_d{d}_r{r}.npz
│
├── 03_pretrain/                    # Stage 3: 预训练输出
│   ├── pretrained_model.pth        # 🔥 预训练模型权重
│   └── pretrain_history.json       # 训练损失/准确率历史
│
├── 04_finetune/                    # Stage 4: 微调输出
│   └── <experiment_name>/          # 每个实验一个子目录
│       └── finetuned_model.pth     # 🔥 微调后的模型权重
│
├── 05_test/                        # Stage 5: 测试输出
│   └── test_results.json           # 📊 所有测试结果 (LER, Accuracy)
│
├── 06_report/                      # Stage 6: 研究报告
│   ├── research_report.md          # 📝 最终研究报告 (Markdown)
│   └── full_metrics.json           # 完整执行指标
│
├── config.json                     # 运行配置参数
├── pipeline_metrics.json           # 执行时间和状态
└── complete_pipeline.log           # 完整执行日志
```

**单独运行对齐检查输出**:

```
alignment_results/
├── alignment_report.json           # 详细检查结果
└── alignment_report.md             # 人类可读报告
```

#### 7.5 关键输出文件说明

| 文件 | 路径 | 说明 |
|------|------|------|
| **预训练模型** | `<output>/03_pretrain/pretrained_model.pth` | 8.5M 样本训练的基础模型 |
| **微调模型** | `<output>/04_finetune/*/finetuned_model.pth` | 针对特定实验配置微调的模型 |
| **测试结果** | `<output>/05_test/test_results.json` | 包含所有模型的 LER 和准确率 |
| **研究报告** | `<output>/06_report/research_report.md` | 完整的论文复现研究报告 |
| **对齐报告** | `<output>/01_alignment/alignment_report.md` | 与论文参数的对齐验证报告 |

#### 7.6 示例：快速测试后的典型输出

运行 `python run_complete_server_pipeline.py --quick-test --output-dir test_results` 后：

```
test_results/
├── 01_alignment/
│   ├── alignment_report.json       # 10 项检查结果
│   └── alignment_report.md
├── 02_data/
│   ├── si1000/
│   │   ├── si1000_d3_p0p005.npz   # ~1000 样本
│   │   └── si1000_d3_p0p010.npz
│   ├── pauli_plus/
│   │   └── samples_surface_code_bX_d3_r05.npz  # ~2000 样本
│   └── test/
│       └── test_bX_d3_r05.npz     # ~500 样本
├── 03_pretrain/
│   ├── pretrained_model.pth        # ~33MB (8M 参数)
│   └── pretrain_history.json
├── 04_finetune/
│   └── samples_surface_code_bX_d3_r05/
│       └── finetuned_model.pth
├── 05_test/
│   └── test_results.json           # {"results": [...]}
├── 06_report/
│   ├── research_report.md          # ~5KB 报告
│   └── full_metrics.json
├── config.json
├── pipeline_metrics.json
└── complete_pipeline.log
```

### 8. 代码文件对应关系

| 论文章节 | 代码文件 |
|---------|---------|
| Model Architecture | `ai_models/model.py`, `ai_models/model_mla.py` |
| Noise Model | `my_noise_model/paper_aligned.py`, `my_noise_model/channels.py` |
| GPTA | `my_noise_model/gpta.py` |
| Soft Readout | `my_noise_model/iq_readout.py`, `my_noise_model/softxor.py` |
| Training | `ai_models/train.py`, `ai_models/model_mla.py` |
| Fine-tuning | `ai_models/fine_tune.py`, `ai_models/fine_tune_npz.py` |
| Table S4 | `my_noise_model/paper_aligned.py` |
| SI1000 | `simulator/si1000_generator.py` |
| Figure 2 | `paper_figures/fig2_threshold_plot.py` |
| Figure 3 | `paper_figures/fig3_decoder_comparison.py` |
| Figure 4 | `paper_figures/fig4_finetuning_results.py` |

### 9. 验证检查清单

运行 `python run_paper_alignment_check.py` 可自动验证以下项目：

- [x] Table S4 所有 10 个噪声参数完全匹配
- [x] 所有 Kraus 通道满足 CPTP 条件
- [x] 模型架构: 12 层, 8 头, 256 维
- [x] 预训练: 8.5M 样本, batch=256, lr=1e-4, epochs=100
- [x] 微调: 50K 样本/实验, batch=128, lr=1e-5, epochs=30
- [x] 数据分割: 80/20 训练/验证
- [x] Soft XOR 公式: p + q - 2pq
- [x] GPTA 实现正确（PTM 对角元 + Hadamard 变换）
- [x] DQLR 矩阵为有效随机矩阵
- [x] SI1000 p 网格: 0.001-0.01 共 10 个值
- [x] 码距: d=3, 5, 7


