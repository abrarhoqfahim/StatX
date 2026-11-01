# Research Data Analysis

A reproducible Python toolkit for exploring research data. The project currently provides command-line tools for descriptive statistics and correlations, with a long-term goal of making reliable analysis accessible and clearly explained.

## Requirements

- Python 3.8 or newer
- No third-party packages

Run commands from the project root. Use `python` instead of `python3` if that is how Python is named on your system.

## Correlation analysis

The correlation tool calculates Pearson, Spearman, and Kendall tau-b coefficients for two numeric columns. It also calculates point-biserial correlation when one column is binary and the other has more than two distinct values.

Run the analysis included with this project:

```bash
python3 scripts/correlation_analysis.py data/pcm_thermal_analysis.csv "2300(MF)" "2300(HST)"
```

The script writes results to `data/pcm_thermal_analysis_correlations.csv` by default. To choose another output path, add `--output path/to/results.csv`.

### Results

Correlation between `2300(MF)` and `2300(HST)` in `data/pcm_thermal_analysis.csv`:

| Method | Coefficient | Complete pairs | Rows skipped |
|---|---:|---:|---:|
| Pearson | 0.7544 | 1,201 | 0 |
| Spearman | 0.9973 | 1,201 | 0 |
| Kendall tau-b | 0.9816 | 1,201 | 0 |
| Point-biserial | Not applicable | 1,201 | 0 |

The results show a strong positive rank association (Spearman and Kendall) and a positive linear association (Pearson) for these columns. Point-biserial correlation does not apply because the selected columns do not meet the required binary/continuous combination. Correlation describes association, not causation; interpret coefficients in light of the variables, measurement scales, and study design.

The script excludes a row if either selected cell is blank and reports the number of complete pairs and skipped rows. Non-numeric or non-finite values, malformed CSV rows, and missing or duplicated column headers produce an error. Undefined or inapplicable coefficients are recorded with a status and note in the results CSV.

To analyze a different pair, replace the input path and exact column names:

```bash
python3 scripts/correlation_analysis.py path/to/data.csv column_x column_y
```

Quote paths or column names containing spaces or shell-special characters. Use `--output` to set a custom results path.

## Descriptive statistics

[`scripts/descriptive_statistics.py`](./scripts/descriptive_statistics.py) reports the count, mean, population variance and standard deviation, and sample variance and standard deviation for one numeric CSV column.

```bash
python3 scripts/descriptive_statistics.py data/pcm_thermal_analysis.csv "2300(MF)"
```

Use the exact CSV header as the column argument. The script skips blank cells and reports their count. Sample variance and standard deviation require at least two numeric values.

## Project direction

This project aims to grow into a beginner-friendly, reproducible toolkit for research data analysis. Planned directions include data validation and visualization, additional statistical methods, and foundational machine-learning workflows. Each addition should make its assumptions transparent and help users interpret results responsibly.
