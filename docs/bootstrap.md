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

Let the observed data be $x_1, x_2, \ldots, x_n$ and the sample mean be:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

For each bootstrap replicate $b = 1, \ldots, B$, draw $n$ observations with
replacement from the original sample and compute the resampled mean:

$$
\bar{x}_b^* = \frac{1}{n}\sum_{j=1}^{n} x_{b,j}^*
$$

The bootstrap estimate of the mean is the average of the resampled means:

$$
\bar{x}_{\text{boot}} = \frac{1}{B}\sum_{b=1}^{B}\bar{x}_b^*
$$

The bootstrap standard error is:

$$
SE_{\text{boot}} =
\sqrt{\frac{1}{B-1}\sum_{b=1}^{B}
\left(\bar{x}_b^*-\bar{x}_{\text{boot}}\right)^2}
$$

For a confidence level $1-\alpha$, the percentile bootstrap confidence interval
is:

$$
\left[
q_{\alpha/2}\left(\bar{x}_b^*\right),
q_{1-\alpha/2}\left(\bar{x}_b^*\right)
\right]
$$

where $q_p$ is the empirical $p$-quantile of the bootstrap distribution. In
the script, percentiles are linearly interpolated between adjacent ordered
bootstrap means.

This approach makes no parametric distributional assumption about the population mean beyond the observed sample being representative of the data-generating process.
