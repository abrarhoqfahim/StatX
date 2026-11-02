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
statistics, and coefficient inference are printed to standard output.
