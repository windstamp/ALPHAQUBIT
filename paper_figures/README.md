# AlphaQubit Paper Figures & Tables

This folder contains scripts to generate all figures and tables from the AlphaQubit paper:

**"Accurate neural network decoding of surface codes for quantum error correction"**  
*Nature, 2024* ([DOI: 10.1038/s41586-024-08449-y](https://doi.org/10.1038/s41586-024-08449-y))

## Quick Start

Generate all figures and tables with a single command:

```bash
python -m paper_figures.generate_all
```

Or with interactive display:

```bash
python -m paper_figures.generate_all --show
```

## Output

All generated files are saved to `paper_figures/output/`:

### Figures

| File | Description |
|------|-------------|
| `fig2_threshold_plot.png` | Threshold and scaling behavior (main) |
| `fig2_threshold_comparison.png` | AlphaQubit vs MWPM threshold comparison |
| `fig3_decoder_comparison.png` | Decoder comparison bar charts |
| `fig3_improvement_chart.png` | AlphaQubit improvement over other decoders |
| `fig3_radar_chart.png` | Multi-metric decoder comparison |
| `fig4_finetuning_results.png` | Fine-tuning results on Google QEC |
| `fig4_grouped_analysis.png` | LER grouped by distance and noise |
| `fig4_heatmap.png` | LER heatmap across configurations |
| `ext_ablation_*.png` | Extended data ablation studies |

### Tables

| File | Description |
|------|-------------|
| `table1_architecture.txt` | Model architecture configurations |
| `table2_training.txt` | Training hyperparameters |
| `table3_decoder_comparison.txt` | Decoder performance comparison |
| `table4_finetuning.txt` | Fine-tuning results summary |

## Individual Scripts

You can also run individual figure generators:

```bash
# Figure 2: Threshold plots
python -m paper_figures.fig2_threshold_plot

# Figure 3: Decoder comparison
python -m paper_figures.fig3_decoder_comparison

# Figure 4: Fine-tuning results
python -m paper_figures.fig4_finetuning_results

# Extended Data: Ablation studies
python -m paper_figures.extended_ablations

# Tables only
python -m paper_figures.generate_tables_simple
```

## Reference Data

The `paper_data.py` module contains all reference values from the paper:

- **Threshold data**: LER vs physical error rate for d=3,5,7
- **Decoder comparison**: AlphaQubit, MWPM, Tensor Network, BP, UF
- **Fine-tuning results**: Google QEC v3.5 experiment results
- **Ablation studies**: Soft readout, pre-training, model size
- **Model architecture**: Configurations from small to xlarge
- **Training config**: Pre-training and fine-tuning hyperparameters

## Customization

### Using Your Own Results

To compare your results with the paper:

1. Save your results to a JSON file
2. Modify `paper_data.py` or pass data directly to plotting functions
3. Run the figure generators

Example:
```python
from paper_figures.fig4_finetuning_results import generate_figure4

# Use custom output directory
generate_figure4(output_dir="my_results/", show=True)
```

### Styling

All figures use `seaborn-v0_8-whitegrid` style. To change:

```python
import matplotlib.pyplot as plt
plt.style.use('your_style')
```

## Dependencies

- `numpy`
- `matplotlib`
- `pathlib` (standard library)

Optional:
- `tabulate` (for formatted table output)
