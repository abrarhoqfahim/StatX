# StatX

StatX is an under-development Python toolkit for clear, reproducible statistical
analysis of research data. Its current tools work primarily with CSV files and
provide descriptive summaries, association analysis, regression, resampling,
and ANOVA.

## Repository map

```text
StatX/
├── Anova/                         # One-way and two-way ANOVA
│   ├── anova_analysis.md                  # ANOVA setup, usage, outputs, and safeguards
│   ├── __init__.py
│   ├── anova_analysis.py          # ANOVA command-line and interactive entry point
│   ├── anova_toolkit/             # Reusable ANOVA implementation
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── data_loader.py
│   │   ├── data_validator.py
│   │   ├── one_way_anova.py
│   │   ├── preprocessing.py
│   │   ├── reporting.py
│   │   ├── results.py
│   │   ├── statistics.py
│   │   ├── two_way_anova.py
│   │   └── visualizations.py
│   └── tests/
│       └── test_anova_toolkit.py
├── docs/                          # Methods and interpretation notes
│   ├── anova.md
│   ├── bootstrap.md
│   ├── correlation_and_descriptive.md
│   └── regression.md
├── regression/                    # OLS regression analysis
│   ├── diagnostic_plots.py        # Regression diagnostic plotting
│   ├── regression_analysis.py     # Regression command-line analysis
│   └── tests/
│       └── test_regression_analysis.py
├── scripts/                       # Standalone CSV analysis tools
│   ├── bootstrap_mean.py
│   ├── correlation_analysis.py
│   └── descriptive_statistics.py
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt               # Shared Python dependencies
```

Generated Python cache files and other ignored local artifacts are omitted
from this map.

## Available analyses

1. **Descriptive statistics** — count, mean, and population and sample
   variance and standard deviation for a numeric CSV column:
   [`scripts/descriptive_statistics.py`](./scripts/descriptive_statistics.py);
   [methodology and details](./docs/correlation_and_descriptive.md).
2. **Correlation** — Pearson, Spearman, Kendall tau-b, and applicable
   point-biserial coefficients for two numeric CSV columns:
   [`scripts/correlation_analysis.py`](./scripts/correlation_analysis.py);
   [methodology and details](./docs/correlation_and_descriptive.md).
3. **Bootstrap mean estimation** — bootstrap standard error and percentile
   confidence interval for a numeric CSV column:
   [`scripts/bootstrap_mean.py`](./scripts/bootstrap_mean.py);
   [methodology and details](./docs/bootstrap.md).
4. **Regression** — simple linear, multiple linear, and polynomial ordinary
   least-squares models, with fit statistics, coefficient inference, and
   diagnostic plots:
   [`regression/regression_analysis.py`](./regression/regression_analysis.py)
   and [`regression/diagnostic_plots.py`](./regression/diagnostic_plots.py);
   [methodology and details](./docs/regression.md).
5. **ANOVA** — one-way classical, Welch, or Kruskal–Wallis analysis and
   two-way factorial ANOVA, with assumption checks, post-hoc comparisons,
   visualizations, and reports:
   [`Anova/anova_analysis.py`](./Anova/anova_analysis.py);
   [methodology](./docs/anova.md) and
   [toolkit documentation](./Anova/anova_analysis.md).

## Install dependencies

Install the packages listed in `requirements.txt` from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Ambition and stage

StatX aims to grow into a dependable, accessible toolkit that makes research
analysis and its assumptions easier to understand. It is currently under
development; methods and interfaces may evolve, and results should be checked
against the relevant methodology and study design.
