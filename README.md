# Research Data Analysis

A Python and notebook workspace for making research data analysis more accessible, transparent, and reproducible. The project currently includes a command-line tool for calculating descriptive statistics from a CSV column.

## Current tool: descriptive statistics

[`descriptive_statistics.py`](./scripts/descriptive_statistics.py) calculates the mean, variance, and standard deviation for one numeric column in a CSV file. It reports both population and sample variance and standard deviation so you can select the interpretation that matches your data.

### Requirements

- Python 3.8 or newer
- No third-party packages are required
- A CSV file with a header row and the requested column

### Run the script

From this project folder, run:

```bash
python descriptive_statistics.py path/to/data.csv column_name
```

For example:

```bash
python descriptive_statistics.py data/study.csv height_cm
```

On systems where Python 3 is invoked as `python3`, use:

```bash
python3 descriptive_statistics.py data/study.csv height_cm
```

The column name must exactly match a CSV header. Quote a path or column name if it contains spaces:

```bash
python descriptive_statistics.py "data/study results.csv" "height cm"
```

### Analysis workflow

1. Prepare a CSV file with a header row and one row per observation.
2. Identify the exact header of the numeric variable to analyze.
3. Run the command with the CSV path followed by that column name.
4. Review the reported count of numeric values and blank cells.
5. Choose the population or sample results that fit the study design, and interpret the results in the context of the research question and measurement units.

The script skips blank cells and reports how many were skipped. It stops with a clear error if the requested column is missing or duplicated, if a nonblank value is not numeric, if a value is infinite or not a number, or if the CSV structure is malformed. At least one numeric value is required. With one numeric value, population statistics are reported; sample variance and standard deviation are undefined and are identified as such.

## Understanding the results

- **Mean** is the arithmetic average of the numeric observations.
- **Population variance** divides the sum of squared deviations from the mean by `n`. Use it when the data represent the entire population of interest.
- **Population standard deviation** is the square root of population variance.
- **Sample variance** divides the sum of squared deviations by `n - 1`. It is commonly used to estimate population variance from a sample and requires at least two observations.
- **Sample standard deviation** is the square root of sample variance.

Variance is expressed in squared units; standard deviation uses the original measurement units. The script describes the data but does not assess study-design assumptions, missing-data mechanisms, outliers, or whether the results support a causal conclusion.

## Project vision

The goal is to grow this repository into a practical, beginner-friendly toolkit for research data analysis using Python scripts and notebooks. Future analyses may cover data checks and visualization, additional descriptive and inferential statistics, and foundational machine-learning workflows. Each addition should make its assumptions clear, preserve reproducibility, and explain results in the context of the research question. These are future goals, not features of the current script.

## Current project files

- `descriptive_statistics.py` — CSV-column descriptive statistics tool.
- `prompt.md` — reusable guidance for accurate, reproducible research-data analysis.
- `.gitignore` — excludes common Python-generated files, local environments, and editor/system files.
