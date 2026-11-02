# ANOVA methodology

This note describes the methods implemented in the StatX ANOVA toolkit. It is
intended to make the model, estimands, and assumptions clear; a test result does
not replace design-specific statistical judgment.

## Data preparation and analysis unit

Each row must represent an independent experimental unit. The CLI requires
explicit confirmation of independence; repeated-measures, clustered, and other
dependent observations require a different model and are not supported here.

For each response, the toolkit selects that response and its factor columns,
converts blank strings to missing values, and uses complete cases. The number of
excluded rows is reported separately for each response. Invalid non-numeric or
infinite response values are rejected rather than silently discarded or
imputed. No outlier removal or imputation is performed.

Let $y_{ij}$ denote observation $j$ in group $i$, with $k$ groups and
$N$ complete observations. The group mean and sample standard deviation
reported in summaries are

$$
\bar y_i = \frac{1}{n_i}\sum_{j=1}^{n_i} y_{ij},
\qquad
s_i = \sqrt{\frac{1}{n_i-1}\sum_{j=1}^{n_i}(y_{ij}-\bar y_i)^2}.
$$

Group summaries also include standard errors and pointwise, two-sided
$t$-based confidence intervals:

$$
\bar y_i \ \pm\ t_{1-\alpha/2,\,n_i-1}\frac{s_i}{\sqrt{n_i}}.
$$

These intervals are not multiplicity-adjusted.

## One-way independent-groups analyses

Choose one method using `--method`. The default is classical ANOVA. The
alternatives are explicit choices; diagnostics do not automatically switch the
test.

### Classical one-way ANOVA

The model is

$$
y_{ij} = \mu + \tau_i + \varepsilon_{ij},
$$

where $\mu$ is the overall level, $\tau_i$ is the group effect, and
$\varepsilon_{ij}$ is residual error. The implementation fits an ordinary
least-squares model with sum-to-zero factor contrasts and obtains a Type II
ANOVA table. For this one-factor model, the factor test is the usual
between-groups versus within-groups F test:

$$
SS_{\text{between}} = \sum_{i=1}^{k} n_i(\bar y_i-\bar y)^2,
\qquad
SS_{\text{error}} = \sum_{i=1}^{k}\sum_{j=1}^{n_i}(y_{ij}-\bar y_i)^2,
$$

$$
F = \frac{SS_{\text{between}}/(k-1)}
         {SS_{\text{error}}/(N-k)}.
$$

Under the null hypothesis that all group means are equal, this statistic is
compared with an $F_{k-1,N-k}$ distribution.

### Welch one-way ANOVA

Welch's test is used when unequal group variances are a concern. For each group,
let $s_i^2$ be its sample variance and define

$$
w_i = \frac{n_i}{s_i^2}, \qquad
\bar y_w = \frac{\sum_i w_i\bar y_i}{\sum_i w_i}, \qquad
C = \sum_i\frac{(1-w_i/\sum_r w_r)^2}{n_i-1}.
$$

The implemented statistic and approximate degrees of freedom are

$$
F_W =
\frac{\sum_i w_i(\bar y_i-\bar y_w)^2/(k-1)}
     {1+\frac{2(k-2)}{k^2-1}C},
\qquad
\nu_1=k-1,\qquad
\nu_2=\frac{k^2-1}{3C}.
$$

The p-value is the upper-tail probability from $F_{\nu_1,\nu_2}$. The
implementation requires at least two observations and nonzero sample variance
in every group.

### Kruskal-Wallis

Kruskal-Wallis is a rank-based independent-group alternative. Pool all
observations, assign average ranks to ties, and let $R_i$ be the rank sum in
group $i$. The test statistic is

$$
H = \frac{12}{N(N+1)}\sum_{i=1}^{k}\frac{R_i^2}{n_i} - 3(N+1).
$$

The SciPy implementation applies tie correction and evaluates the large-sample
reference distribution with $k-1$ degrees of freedom. This test concerns
group rank distributions; interpreting it strictly as a test of medians
requires similarly shaped group distributions. The reported rank-effect
estimate is

$$
\epsilon^2 = \max\left(0,\frac{H-k+1}{N-k}\right).
$$

Each group must have at least two observations.

## Two-way factorial ANOVA

The two-way model includes both main effects and their interaction:

$$
y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij}
          + \varepsilon_{ijk},
$$

where $\alpha_i$ and $\beta_j$ are the effects of factors A and B, and
$(\alpha\beta)_{ij}$ is their interaction. The implementation fits this model
by ordinary least squares using sum-to-zero contrasts. Every combination of
factor levels must contain at least one complete observation; empty cells are
rejected because factorial effects can be non-estimable.

The requested sums-of-squares type (`--sum-of-squares I`, `II`, or `III`) is
passed to the ANOVA table calculation:

- **Type I** tests terms sequentially in model order.
- **Type II** tests each main effect after the other main effect, and tests the
  interaction after the main effects.
- **Type III** tests each term conditional on all other terms in the model. The
  toolkit uses sum-to-zero contrasts, which are important for interpreting
  Type III tests.

For a term $q$, the conventional test statistic is

$$
F_q = \frac{SS_q/df_q}{SS_{\text{error}}/df_{\text{error}}},
$$

with its p-value evaluated against the corresponding F reference distribution.
In an unbalanced design, sums-of-squares types can give different tests;
report the chosen type and interpret main effects in light of any interaction.

## Effect sizes

For each estimable ANOVA term $q$, the toolkit reports eta-squared, partial
eta-squared, and omega-squared:

$$
\eta^2_q = \frac{SS_q}{SS_{\text{total}}},
\qquad
\eta^2_{p,q} = \frac{SS_q}{SS_q+SS_{\text{error}}},
$$

$$
\omega^2_q =
\max\left(0,\frac{SS_q-df_q\,MSE}{SS_{\text{total}}+MSE}\right),
\qquad
MSE = \frac{SS_{\text{error}}}{df_{\text{error}}}.
$$

For the classical one-way analysis, the reported effect in the interpretation
is eta-squared. For the two-way analysis, the per-term interpretation reports
partial eta-squared. Effect-size estimates describe magnitude on this model and
design; they do not establish practical importance.

## Diagnostics and post-hoc comparisons

For classical one-way and two-way models, the toolkit reports:

- **Shapiro-Wilk test** of model residuals when there are 3–5,000
  non-constant residuals.
- **Median-centered Levene test** for equal variances across the supplied
  groups (one-way groups or two-way cells), when each group has at least two
  observations and the statistic is defined.

These diagnostics do not test independence, prove assumptions, or establish
that a model is appropriate. Independence must be justified by the study
design. Welch and Kruskal-Wallis are available alternatives for one-way
analyses; a diagnostic p-value does not select them automatically. The
Kruskal-Wallis path marks parametric residual/variance diagnostics as not
applicable.

Post-hoc procedures are:

- Classical one-way: Tukey HSD pairwise comparisons and familywise confidence
  intervals.
- Welch one-way: Games-Howell pairwise comparisons and confidence intervals.
- Kruskal-Wallis: two-sided pairwise Mann-Whitney tests with Holm-adjusted
  p-values. The displayed difference is the difference in sample medians; no
  confidence interval is claimed.
- Two-way: if the interaction p-value is below $\alpha$, Tukey comparisons
  are performed for simple effects within levels of the other factor.
  Otherwise, Tukey comparisons are produced for main effects whose p-values
  are below $\alpha$.

Use the interaction plot and simple effects to interpret a significant
interaction rather than relying on isolated marginal effects. Statistical
significance does not establish causation; conclusions depend on the sampling
and experimental design, data quality, model assumptions, and scientific
context.

## Running the analysis

From the project root, a one-way analysis can be run with:

```bash
python Anova/anova_analysis.py --input path/to/data.csv --responses outcome --factor treatment --method classical --confirm-independent-observations
```

A two-way analysis can be run with:

```bash
python Anova/anova_analysis.py --input path/to/data.csv --responses outcome --factor material --factor-b temperature --analysis two-way --sum-of-squares II --confirm-independent-observations
```

See [Anova/README.md](../Anova/README.md) for setup, output files, and additional
CLI options.
