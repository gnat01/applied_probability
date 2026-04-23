# How to Run

This directory contains three typo-probability experiments:

1. The original independent-page model.
2. A Markovian tired-typist model that creates bursty typo behavior.
3. A Hawkes-like self-exciting point process model.

Directory layout:

```text
src/      Python scripts
outputs/  Generated CSV and PNG files
docs/     Explanatory notes and documentation images
```

## Requirements

The scripts use:

```bash
python
numpy
matplotlib
scipy
```

`scipy` is only needed by `typo_crossover_vs_alpha_full_pipeline.py`.

## Independent-page model

From this directory:

```bash
python src/typo_crossover_vs_alpha_full_pipeline.py
```

From the repository root:

```bash
python typos_on_a_page/src/typo_crossover_vs_alpha_full_pipeline.py
```

By default, the independent full pipeline writes generated files into:

```text
typos_on_a_page/outputs/
```

To choose a different output directory:

```bash
python typos_on_a_page/src/typo_crossover_vs_alpha_full_pipeline.py --output-dir /tmp/independent-typos
```

## Markovian tired-typist model

From this directory:

```bash
python src/markovian_tired_typist.py
```

From the repository root:

```bash
python typos_on_a_page/src/markovian_tired_typist.py
```

By default, the Markovian script writes generated files into:

```text
typos_on_a_page/outputs/
```

To choose a different output directory:

```bash
python typos_on_a_page/src/markovian_tired_typist.py --output-dir /tmp/markovian-typos
```

## Hawkes-like point-process model

From this directory:

```bash
python src/hawkes_typo_process.py
```

From the repository root:

```bash
python typos_on_a_page/src/hawkes_typo_process.py
```

By default, the Hawkes script writes generated files into:

```text
typos_on_a_page/outputs/
```

To choose a different output directory:

```bash
python typos_on_a_page/src/hawkes_typo_process.py --output-dir /tmp/hawkes-typos
```

To change the Monte Carlo size or seed:

```bash
python typos_on_a_page/src/hawkes_typo_process.py --simulations 12000 --seed 19
```

If Matplotlib warns that `~/.matplotlib` is not writable, the run can still
complete. To avoid the warning and speed up future runs, use:

```bash
MPLCONFIGDIR=.mplconfig python typos_on_a_page/src/hawkes_typo_process.py --output-dir typos_on_a_page/outputs
```

Generated outputs:

```text
outputs/crossover_values.csv
outputs/fit_parameters.csv
outputs/crossover_curves.png
outputs/crossover_curves_with_fits.png
outputs/typo_crossover_vs_alpha.png
outputs/typo_crossover_vs_alpha_multi_threshold.png
outputs/markovian_crossover_values.csv
outputs/markovian_probability_curves_mild.png
outputs/markovian_probability_curves_baseline.png
outputs/markovian_probability_curves_fragile.png
outputs/markovian_sample_bursts_mild.png
outputs/markovian_sample_bursts_baseline.png
outputs/markovian_sample_bursts_fragile.png
outputs/markovian_crossover_lines_mild.png
outputs/markovian_crossover_lines_baseline.png
outputs/markovian_crossover_lines_fragile.png
outputs/markovian_crossover_heatmap_mild.png
outputs/markovian_crossover_heatmap_baseline.png
outputs/markovian_crossover_heatmap_fragile.png
outputs/markovian_stationary_typo_rate.png
outputs/hawkes_comparison_summary.csv
outputs/hawkes_crossover_comparison.csv
outputs/hawkes_probability_curves.png
outputs/hawkes_model_comparison_curves.png
outputs/hawkes_inter_typo_gap_comparison.png
outputs/hawkes_burst_size_comparison.png
outputs/hawkes_crossover_comparison.png
outputs/hawkes_sample_path_mild.png
outputs/hawkes_sample_path_baseline.png
outputs/hawkes_sample_path_volatile.png
```

## What to inspect first

Start with:

```text
outputs/markovian_probability_curves_baseline.png
outputs/markovian_sample_bursts_baseline.png
outputs/markovian_crossover_lines_baseline.png
outputs/markovian_crossover_heatmap_baseline.png
outputs/hawkes_model_comparison_curves.png
outputs/hawkes_inter_typo_gap_comparison.png
outputs/hawkes_burst_size_comparison.png
```

These show how increasing tired-state persistence changes the cumulative
probability of seeing at least one typo and the crossover page count.
