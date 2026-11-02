# Regression methodology

This note describes the OLS models and statistics reported by the regression
command. Regression estimates conditional associations; it does not by itself
establish causation.

## Data preparation

Each CSV row is treated as one observation. The target and all selected
predictors must be numeric. Rows with a blank value in any selected column are
excluded as complete cases and counted in the output. A non-numeric or
non-finite selected value, malformed row, missing column, or duplicate header
stops the analysis; values are not imputed or silently coerced. The model
requires more complete observations than fitted coefficients, and its design
matrix must have full column rank.

The command reports conventional, non-robust OLS standard errors, two-sided
coefficient t tests, and pointwise confidence intervals. Their usual inferential
interpretation relies on a correctly specified mean model, independent errors,
constant error variance, and (for exact small-sample t/F inference) normally
distributed errors. The tool does not diagnose these assumptions or adjust
inference for multiple testing.

## Models

Let $y_i$ be the response for observation $i$, and let $\varepsilon_i$ be the
error term.

### Simple linear regression

Exactly one predictor $x$ is allowed:

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i.
$$

### Multiple linear regression

At least two predictors are required. The model includes one linear term per
predictor and an intercept; interactions are not included:

$$
y_i = \beta_0 + \sum_{j=1}^{p}\beta_j x_{ij} + \varepsilon_i.
$$

### Polynomial regression

Exactly one predictor is allowed. The default degree is two; `--degree` may
select any integer degree of at least two:

$$
y_i = \beta_0 + \beta_1 z_i + \beta_2 z_i^2 + \cdots
      + \beta_d z_i^d + \varepsilon_i,
\qquad
z_i = \frac{x_i-\bar{x}}{s_x},
$$

where $\bar{x}$ is the sample mean and $s_x$ is the population standard
deviation of the complete predictor values. Centering and scaling improve the
numerical conditioning of polynomial powers. The output shows this basis
transformation; polynomial coefficients are expressed in $z$, not in the
original units of $x$. This model does not add interactions or select the
degree automatically.

## Estimation and reported statistics

For design matrix $X$ and response vector $y$, ordinary least squares estimates
the coefficients by minimizing the residual sum of squares:

$$
\hat{\boldsymbol{\beta}}
= \arg\min_{\boldsymbol{\beta}}\|y-X\boldsymbol{\beta}\|_2^2
= (X^\mathsf{T}X)^{-1}X^\mathsf{T}y
$$

when the design has full column rank. For $n$ observations and $k$ fitted
coefficients (including the intercept), the output includes:

- **Coefficient estimates**, conventional standard errors, t statistics,
  two-sided p-values, and pointwise confidence intervals at the selected
  confidence level. The intervals use the Student t distribution with $n-k$
  residual degrees of freedom.
- **$R^2$** and **adjusted $R^2$**, summarizing in-sample fit and fit adjusted
  for the number of estimated parameters.
- **Model F statistic and p-value**, testing the joint null that all
  non-intercept coefficients are zero.
- **RMSE**, $\sqrt{\mathrm{SSE}/n}$, and **residual standard error**,
  $\sqrt{\mathrm{SSE}/(n-k)}$.
- **AIC** and **BIC**, likelihood-based criteria that can help compare models
  fit to the same response observations; lower values indicate a preferred
  tradeoff among compared candidates.

These are in-sample estimates. High $R^2$ is not proof of predictive
performance, correct assumptions, or causation. Use design knowledge and
out-of-sample validation where prediction is the goal.

## Evaluation on the included example datasets

The following checks used the CSV files in `data/`. Cross-validation scores
are calculated from held-out predictions, not from the full-data fit. Reported
$R^2$, RMSE, and MAE are pooled across all out-of-fold predictions. RMSE and
MAE for concrete strength are in MPa.

### Synthetic polynomial example

`data/ideal_data.csv` has 101 data rows, with $x$ values from 0 to 10 in
increments of 0.1. This observed range includes 0, despite a 1-to-10 range
being intended. Two rows have blank responses ($x=0.9$ and $x=2$), leaving 99
complete observations. The complete values follow

$$
y = 5 + 3x + 2x^2
$$

to a maximum absolute discrepancy of approximately $5.7\times10^{-14}$,
consistent with floating-point rounding. On a shuffled, seeded five-fold
validation split (seed 42), the models scored:

| Model | Cross-validated $R^2$ | RMSE | MAE |
|---|---:|---:|---:|
| Linear, degree 1 | 0.9478 | 15.6816 | 13.4921 |
| Polynomial, degree 2 | 1.0000 | $1.7\times10^{-13}$ | $9.6\times10^{-14}$ |
| Polynomial, degree 3 | 1.0000 | $8.4\times10^{-14}$ | $5.7\times10^{-14}$ |

The quadratic is the appropriate model here: it recovers the known relationship
to numerical precision, while degree three provides no meaningful improvement
and adds an unnecessary term. Because this is a deterministic synthetic
relationship with effectively zero residual variance, its conventional
standard errors and p-values are numerically degenerate and should not be
interpreted as evidence about a noisy real-world process.

Reproduce the full-data quadratic fit and save its result CSV with:

```bash
python regression/regression_analysis.py data/ideal_data.csv --target y --predictors x --model polynomial --degree 2 --output data/ideal_data_poly2_results.csv
```

### Concrete compressive strength

`data/Concrete_Data.csv` contains 1,030 complete observations. The target is
`Concrete_compressive_strength` in MPa. The seven ingredient amounts (cement,
blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, and
fine aggregate) are in kg/m$^3$; `Age_day` is measured in days, not kg/m$^3$.
The benchmark compared age-only linear and quadratic models with multiple
linear regression using all eight available predictors.

Five-fold cross-validation used seed 42 and grouped rows by their seven
ingredient amounts, so rows with the same ingredient recipe were kept in the
same fold. This held out unseen recipes while allowing age to remain a
predictor. There were 427 distinct recipes. Results were:

| Model | Predictors | Cross-validated $R^2$ | RMSE (MPa) | MAE (MPa) |
|---|---|---:|---:|---:|
| Linear | Age only | 0.1009 | 15.8329 | 12.6572 |
| Polynomial, degree 2 | Age only | 0.2496 | 14.4645 | 11.6130 |
| Multiple linear | All eight predictors | 0.6082 | 10.4514 | 8.2931 |

A fold-specific training-mean baseline had RMSE 16.7614 MPa. The all-predictor
linear model substantially outperformed the age-only models and that baseline,
but its held-out RMSE of about 10.45 MPa is still a material prediction error.
The full-data multiple-model $R^2$ was 0.6155 and in-sample RMSE was 10.3536
MPa; these in-sample numbers are not a substitute for the grouped validation
scores. This benchmark supports the multiple-linear model as a useful baseline,
not as a complete strength-design or engineering acceptance method. The model
does not include interactions or nonlinear terms, and results may vary with
validation design and data quality.

The full-data fits and diagnostic plots generated for this evaluation are:

- Ideal polynomial fit: [`ideal_data_poly2_results.csv`](../data/ideal_data_poly2_results.csv) and [`ideal_data_polynomial_diagnostics.png`](../data/ideal_data_polynomial_diagnostics.png).
- Concrete age-only linear fit: [`Concrete_Data_age_linear_results.csv`](../data/Concrete_Data_age_linear_results.csv) and [`Concrete_Data_linear_diagnostics.png`](../data/Concrete_Data_linear_diagnostics.png).
- Concrete age-only quadratic fit: [`Concrete_Data_age_poly2_results.csv`](../data/Concrete_Data_age_poly2_results.csv) and [`Concrete_Data_polynomial_diagnostics.png`](../data/Concrete_Data_polynomial_diagnostics.png).
- Concrete all-predictor multiple fit: [`Concrete_Data_multiple_results.csv`](../data/Concrete_Data_multiple_results.csv) and [`Concrete_Data_multiple_diagnostics.png`](../data/Concrete_Data_multiple_diagnostics.png).

Dataset citation:

```bibtex
@misc{concrete_compressive_strength_165,
  author       = {Yeh, I-Cheng},
  title        = {{Concrete Compressive Strength}},
  year         = {1998},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5PK67}
}
```

## Command

Run from the StatX project root. Provide exact column names as they appear in
the CSV; quote names containing spaces or shell-special characters.

Simple linear regression:

```bash
python regression/regression_analysis.py data.csv --target outcome --predictors dose --model linear
```

Multiple linear regression:

```bash
python regression/regression_analysis.py data.csv --target outcome --predictors dose temperature --model multiple
```

Polynomial regression (degree two by default, or set a different degree):

```bash
python regression/regression_analysis.py data.csv --target outcome --predictors dose --model polynomial --degree 3
```

Use `--confidence-level 0.90` to request 90% coefficient confidence
intervals; the default is 0.95. The fitted equation, sample counts, model fit
statistics, and coefficient inference are printed to standard output and saved
to a CSV file beside the input, named `<input-name>_regression_results.csv`.
Choose a different output path with `--output`:

```bash
python regression/regression_analysis.py data.csv --target outcome --predictors dose --model linear --output results/linear_fit.csv
```

The results CSV contains model metadata and fit statistics as rows, followed
by coefficient rows with estimates, standard errors, t statistics, p-values,
and confidence interval bounds. Its `section` and `metric` columns identify
each row.

Each run also automatically saves a diagnostic-plot image under the project
`data/` directory, named `<input-name>_<model>_diagnostics.png`. It contains
observed-versus-predicted values, residuals versus fitted values, a normal Q-Q
plot, Cook's distance, and residuals versus each predictor. Simple linear and
polynomial models also include observed and fitted response versus the
predictor. Each panel includes a legend identifying its data series or
reference line. The Cook's-distance reference line at $4/n$ is a heuristic,
not a formal cutoff. Plots help identify possible issues; they do not prove
model assumptions.
