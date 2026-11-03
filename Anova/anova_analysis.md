# Anova — Research ANOVA Toolkit

A modular Python toolkit for reproducible one-way and two-way ANOVA on user-provided CSV data. It supports multiple response variables, explicit assumption checks, group summaries, post-hoc comparisons, effect sizes, confidence intervals, diagnostic plots, and CSV, Excel, and PDF reports.

The implementation follows the initial v0.1.0 scope of the supplied development roadmap. Repeated-measures ANOVA, mixed models, MANOVA, ANCOVA, and other future methods are not included.

## Setup

Use Python 3.11 or newer. From the StatX project root, create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows, from the project root:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux, from the project root:

```bash
source .venv/bin/activate
```

Install the shared StatX dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run a one-way ANOVA

Run from the StatX project root. Quote column names with spaces or punctuation:

```bash
python Anova/anova_analysis.py --input path/to/data.csv --responses outcome --factor treatment --analysis one-way --confirm-independent-observations
```

Use `--method classical` (default), `--method welch`, or `--method kruskal` to choose the one-way test. For example:

```bash
python Anova/anova_analysis.py --input data.csv --responses temperature strength --factor material --method welch --confirm-independent-observations --output-dir results
```

## Run a two-way ANOVA

```bash
python Anova/anova_analysis.py --input path/to/data.csv --responses outcome --factor material --factor-b temperature --analysis two-way --sum-of-squares II --confirm-independent-observations
```

Choose `--sum-of-squares I`, `II`, or `III`. The model uses sum-to-zero contrasts, including for Type III tests; the selected model and sums-of-squares type are recorded in the results. Unbalanced designs are supported when every factor combination has observations. Empty cells are rejected because they can make factorial effects non-estimable.

## Interactive use

To enter missing choices through prompts:

```bash
python Anova/anova_analysis.py --interactive
```

The programmatic interface is also available:

```python
from pathlib import Path

from Anova.anova_toolkit import OneWayConfig, analyze_one_way

config = OneWayConfig(
    input_path=Path("data.csv"),
    responses=("outcome",),
    factor_a="treatment",
    independent_observations_confirmed=True,
)
results = analyze_one_way(config)
print(results["outcome"].anova_table)
```

## Results

Each run creates an output directory containing:

- Per-response ANOVA/test, group-summary, assumption, post-hoc, and metadata CSV/JSON files.
- `anova_results.xlsx` with a separate worksheet for each result table.
- `anova_report.pdf` with the analysis tables, interpretations, and figures.
- `run_manifest.json` with the source path, dataset summary, and model metadata.
- `figures/` with 300-DPI distribution, group-mean, and diagnostic plots; two-way analyses also include interaction plots.
- `dataset_summary.json`.

Group means include pointwise t-based confidence intervals. Tukey HSD (or Games–Howell for Welch's ANOVA) provides multiplicity-adjusted pairwise comparisons and intervals. Kruskal–Wallis follow-up comparisons use Holm-adjusted Mann–Whitney tests; no confidence interval is claimed for those comparisons.

## Data and statistical safeguards

- CSV files must have unique headers. Responses must be numeric; selected invalid non-numeric and infinite response values stop analysis with an error.
- Rows missing a selected response or factor are excluded for that response, and the number excluded is reported. No outlier removal or imputation is performed.
- Each row must represent an independent experimental unit. Analysis stops unless the caller confirms independence. Repeated measures and clustered observations require a different model.
- One-way classical ANOVA reports Shapiro–Wilk residual and median-centered Levene diagnostics; Welch and Kruskal–Wallis are explicit alternatives, not silently selected by a diagnostic p-value.
- Two-way ANOVA includes both main effects and their interaction. Type I, II, and III sums of squares are available. When an interaction is significant, post-hoc comparisons are simple effects within the other factor's levels.
- Diagnostics can reveal possible assumption violations; they do not prove assumptions. Statistical significance does not establish causation or practical importance.

## Validation and development

Run the test suite from the StatX project root:

```bash
python -m unittest discover -s Anova/tests -v
```

The tests cover known synthetic designs, independent-group safeguards, missing and invalid values, alternative one-way methods, two-way effects, and report generation. Use a design-appropriate model and consult a statistician for repeated, clustered, or otherwise dependent observations.
