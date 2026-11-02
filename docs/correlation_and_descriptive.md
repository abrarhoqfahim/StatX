# Correlation and descriptive statistics

These command-line tools summarize numeric CSV data. Correlations measure
association, not causation.

## Correlation

Run from the project root using exact CSV column names:

```bash
python3 scripts/correlation_analysis.py data/pcm_thermal_analysis.csv "2300(MF)" "2300(HST)"
```

By default, results are written beside the input as
`<input-name>_correlations.csv`. Specify `--output path/to/results.csv` to
choose another location. Rows with a blank in either selected column are
excluded as complete pairs; invalid numeric values, non-finite values,
malformed rows, and missing or duplicated selected headers produce an error.

For $n$ complete pairs $(x_i,y_i)$, the tool reports:

- **Pearson correlation** measures linear association:

  $$
  r = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
           {\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2
                  \sum_{i=1}^{n}(y_i-\bar{y})^2}}.
  $$

- **Spearman correlation** is Pearson correlation applied to average ranks
  (ties receive their average rank):

  $$
  \rho_s = \operatorname{corr}(R_x,R_y).
  $$

- **Kendall tau-b** accounts for ties. Let $C$ and $D$ be concordant and
  discordant pair counts, and $T_x,T_y$ the counts tied only in $x$ and only
  in $y$. Then

  $$
  \tau_b = \frac{C-D}{\sqrt{(C+D+T_x)(C+D+T_y)}}.
  $$

- **Point-biserial correlation** is reported only when one selected variable
  has exactly two observed values and the other has more than two. It is
  Pearson correlation after coding the binary variable's lower value as 0 and
  higher value as 1:

  $$
  r_{pb} =
  \frac{\bar{y}_1-\bar{y}_0}{s_y}
  \sqrt{\frac{n_0n_1}{n(n-1)}}.
  $$

  Here $s_y$ is the sample standard deviation of the continuous variable,
  and $n_0,n_1$ are the binary-group sizes. If the binary variable is the
  second column, the returned coefficient is still a symmetric correlation
  (with the continuous variable as the other input).

Undefined coefficients (for example, when a required variable has zero
variance) are marked in the result CSV; inapplicable point-biserial results
include a note. Correlation values range from -1 to 1 when defined. Pearson
captures linear association; rank correlations capture monotonic association.
Interpret each in light of distribution, outliers, measurement, and study
design.

The README's example for `data/pcm_thermal_analysis.csv` reports:

| Method | Coefficient | Complete pairs | Rows skipped |
|---|---:|---:|---:|
| Pearson | 0.7544 | 1,201 | 0 |
| Spearman | 0.9973 | 1,201 | 0 |
| Kendall tau-b | 0.9816 | 1,201 | 0 |
| Point-biserial | Not applicable | 1,201 | 0 |

These values describe association only and do not establish a causal
relationship.

## Descriptive statistics

Run the descriptive-statistics command with the input CSV and exact column
name:

```bash
python3 scripts/descriptive_statistics.py data/pcm_thermal_analysis.csv "2300(MF)"
```

Blank cells in the selected column are skipped and counted. Non-numeric or
non-finite values, malformed rows, and missing or duplicated column headers
produce an error. For the $n$ numeric observations $x_1,\ldots,x_n$, the tool
reports the mean, population variance and standard deviation, and (when
$n \ge 2$) sample variance and standard deviation:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i,
\qquad
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2,
\qquad
\sigma = \sqrt{\sigma^2},
$$

$$
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2,
\qquad
s = \sqrt{s^2}.
$$

Population statistics divide by $n$ and describe the observed values as the
entire population of interest. Sample statistics divide by $n-1$ (Bessel's
correction) to estimate population variance from a sample. Sample variance
and standard deviation are undefined when fewer than two numeric values are
available.

See [`README.md`](../README.md) for project setup and additional usage notes.
