"""One-way ANOVA and supported independent-group alternatives."""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from .config import OneWayConfig
from .data_loader import load_csv
from .data_validator import validate_dataset
from .preprocessing import group_descriptives, prepare_response
from .results import AnalysisResult
from .statistics import (
    add_effect_sizes,
    pairwise_mann_whitney,
    pairwise_tukey,
    residual_assumptions,
)


def _formula_name(name: str) -> str:
    return "Q({!r})".format(name)


def _groups(frame: pd.DataFrame, response: str, factor: str) -> List[Tuple[str, np.ndarray]]:
    return [
        (str(level), values.to_numpy(dtype=float))
        for level, values in frame.groupby(factor, observed=True, sort=True)[response]
    ]


def _welch_anova(groups: Sequence[Tuple[str, np.ndarray]]) -> Tuple[float, float, float, float]:
    """Welch's one-way F test with its Satterthwaite denominator df."""
    if len(groups) < 2 or any(len(values) < 2 for _, values in groups):
        raise ValueError("Welch's ANOVA requires at least two observations per group.")
    counts = np.asarray([len(values) for _, values in groups], dtype=float)
    means = np.asarray([np.mean(values) for _, values in groups], dtype=float)
    variances = np.asarray([np.var(values, ddof=1) for _, values in groups], dtype=float)
    if np.any(variances == 0):
        raise ValueError("Welch's ANOVA is undefined for a group with zero variance.")
    weights = counts / variances
    weight_sum = float(np.sum(weights))
    weighted_mean = float(np.sum(weights * means) / weight_sum)
    group_count = len(groups)
    correction = float(
        np.sum((1 - weights / weight_sum) ** 2 / (counts - 1))
    )
    denominator_df = (group_count**2 - 1) / (3 * correction)
    numerator = float(np.sum(weights * (means - weighted_mean) ** 2) / (group_count - 1))
    correction_factor = 1 + (
        2 * (group_count - 2) / (group_count**2 - 1)
    ) * correction
    statistic = numerator / correction_factor
    p_value = float(stats.f.sf(statistic, group_count - 1, denominator_df))
    return statistic, p_value, float(group_count - 1), float(denominator_df)


def _interpretation(
    method: str, statistic: float, p_value: float, alpha: float, effect: float
) -> List[str]:
    result = (
        "The {} test provides evidence of differences among groups (p = {:.4g}; "
        "alpha = {:.3g}).".format(method, p_value, alpha)
        if p_value < alpha
        else "The {} test did not provide sufficient evidence of group differences "
        "(p = {:.4g}; alpha = {:.3g}).".format(method, p_value, alpha)
    )
    return [
        result,
        "The reported effect-size estimate is {:.4g}; practical importance depends "
        "on the research context.".format(effect),
        "Statistical significance does not establish causation or practical importance.",
    ]


def _analyze_response(frame: pd.DataFrame, response: str, config: OneWayConfig) -> AnalysisResult:
    data, excluded = prepare_response(frame, response, (config.factor_a,))
    factor = config.factor_a
    grouped = _groups(data, response, factor)
    if len(grouped) < 2:
        raise ValueError("At least two observed groups are required after missing-data handling.")
    observed_levels = {level for level, _ in grouped}
    missing_levels = [
        str(level)
        for level in data[factor].cat.categories
        if str(level) not in observed_levels
    ]
    if missing_levels:
        raise ValueError(
            "Factor level(s) have no complete response observations: {}.".format(
                ", ".join(missing_levels)
            )
        )
    group_means = {
        level: float(np.mean(values))
        for level, values in data.groupby(factor, observed=True, sort=True)[response]
    }
    fitted = data[factor].map(group_means).to_numpy(dtype=float)
    residuals = data[response].to_numpy(dtype=float) - fitted
    if config.method in {"welch", "kruskal"} and any(len(values) < 2 for _, values in grouped):
        raise ValueError(
            "{} ANOVA requires at least two observations per group.".format(
                "Welch" if config.method == "welch" else "Kruskal-Wallis"
            )
        )

    model_description = "Independent groups; response ~ C({!r}, Sum).".format(factor)
    if config.method == "classical":
        formula = "{} ~ C({}, Sum)".format(_formula_name(response), _formula_name(factor))
        model = ols(formula, data=data).fit()
        table = anova_lm(model, typ=2)
        table.index = [
            "factor" if str(index).startswith("C(") else str(index)
            for index in table.index
        ]
        table = add_effect_sizes(table, "Residual")
        statistic = float(table.loc["factor", "F"])
        p_value = float(table.loc["factor", "PR(>F)"])
        effect = float(table.loc["factor", "eta_squared"])
        assumptions = residual_assumptions(
            residuals, [values for _, values in grouped], config.alpha
        )
        posthoc = pairwise_tukey(grouped, config.alpha)
    elif config.method == "welch":
        statistic, p_value, df_num, df_den = _welch_anova(grouped)
        values = data[response].to_numpy(dtype=float)
        overall_mean = float(np.mean(values))
        total_ss = float(np.sum((values - overall_mean) ** 2))
        between_ss = float(
            sum(
                len(group_values)
                * (float(np.mean(group_values)) - overall_mean) ** 2
                for _, group_values in grouped
            )
        )
        effect = between_ss / total_ss if total_ss > 0 else np.nan
        table = pd.DataFrame(
            [
                {
                    "df": df_num,
                    "F": statistic,
                    "PR(>F)": p_value,
                    "eta_squared": effect,
                    "partial_eta_squared": np.nan,
                    "omega_squared": np.nan,
                }
            ],
            index=["factor"],
        )
        table.loc["Denominator df", "df"] = df_den
        table.loc["Denominator df", ["F", "PR(>F)"]] = np.nan
        assumptions = residual_assumptions(
            residuals,
            [values for _, values in grouped],
            config.alpha,
        )
        posthoc = pairwise_tukey(grouped, config.alpha, equal_var=False)
        model_description += " Welch correction for unequal variances."
    else:
        result = stats.kruskal(*(values for _, values in grouped))
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        table = pd.DataFrame(
            [
                {
                    "df": len(grouped) - 1,
                    "H": statistic,
                    "PR(>H)": p_value,
                    "epsilon_squared": max(
                        0.0, (statistic - len(grouped) + 1) / (len(data) - len(grouped))
                    ),
                }
            ],
            index=["factor"],
        )
        effect = float(table.loc["factor", "epsilon_squared"])
        assumptions = pd.DataFrame(
            [
                {
                    "test": "Parametric residual/variance diagnostics",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "status": "not applicable",
                    "note": "Kruskal-Wallis is a rank-based independent-group alternative.",
                }
            ]
        )
        posthoc = pairwise_mann_whitney(grouped, config.alpha)
        model_description = "Kruskal-Wallis rank test; independent groups."

    summary = group_descriptives(data, response, (factor,), config.alpha)
    interpretation = _interpretation(
        config.method, statistic, p_value, config.alpha, effect
    )
    if config.method == "classical":
        assumption_statuses = set(assumptions["status"])
        if "possible variance heterogeneity" in assumption_statuses:
            interpretation.append(
                "The variance check indicates possible heterogeneity; inspect the "
                "diagnostics and consider Welch's ANOVA."
            )
        if "possible deviation" in assumption_statuses:
            interpretation.append(
                "The residual normality check indicates a possible deviation; inspect "
                "the Q-Q plot and assess whether a rank-based alternative fits the "
                "research question."
            )
    return AnalysisResult(
        response=response,
        analysis="one-way {}".format(config.method),
        complete_n=len(data),
        excluded_n=excluded,
        alpha=config.alpha,
        model_description=model_description,
        group_summary=summary,
        anova_table=table,
        assumptions=assumptions,
        posthoc=posthoc,
        interpretation=interpretation,
        diagnostics={
            "fitted": fitted,
            "residuals": residuals,
            "factor": factor,
            "factor_a": factor,
            "method": config.method,
            "plot_data": data,
        },
    )


def analyze_one_way(config: OneWayConfig) -> Dict[str, AnalysisResult]:
    """Analyze each response column using the selected one-way method."""
    if not config.independent_observations_confirmed:
        raise ValueError(
            "Confirm that rows are independent experimental units before analysis. "
            "Repeated measures require a different model."
        )
    frame = load_csv(config.input_path)
    validate_dataset(frame, config.responses, (config.factor_a,))
    return {
        response: _analyze_response(frame, response, config)
        for response in config.responses
    }
