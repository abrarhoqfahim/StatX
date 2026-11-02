# Bootstrap mean estimation

This repository includes a bootstrap helper for estimating a confidence interval for the mean of a numeric variable from a CSV column.

## Command

From the project root:

```bash
python scripts/bootstrap_mean.py path/to/data.csv column_name --n 10000 --confidence-level 0.95 --seed 42
```

Optional JSON output file:

```bash
python scripts/bootstrap_mean.py path/to/data.csv column_name --n 10000 --confidence-level 0.95 --seed 42 --output results/bootstrap_summary.json
```

The script prints a JSON object with:

- `sample_mean`
- `bootstrap_mean`
- `bootstrap_se`
- `confidence_level`
- `ci_lower`
- `ci_upper`
- `bootstrap_means`

## Mathematical basis

Let the observed data be `x_1, x_2, ..., x_n` and the sample mean be:

```text
x̄ = (1 / n) * Σ x_i
```

For each bootstrap replicate `b = 1, ..., B`, draw `n` observations with replacement from the original sample and compute the resampled mean:

```text
x̄_b* = (1 / n) * Σ x_i*^{(b)}
```

The bootstrap estimate of the mean is the average of the resampled means:

```text
x̄_boot = (1 / B) * Σ x̄_b*
```

The bootstrap standard error is:

```text
SE_boot = sqrt( (1 / (B - 1)) * Σ (x̄_b* - x̄_boot)^2 )
```

For a confidence level `1 - α`, the percentile bootstrap confidence interval is:

```text
[ q_{α/2}(x̄_b*), q_{1 - α/2}(x̄_b*) ]
```

where `q_p` is the empirical percentile of the bootstrap distribution.

This approach makes no parametric distributional assumption about the population mean beyond the observed sample being representative of the data-generating process.
