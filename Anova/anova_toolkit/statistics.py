"""Reusable model diagnostics, effect sizes, and post-hoc comparisons."""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def residual_assumptions(
    residuals: Sequence[float],
    groups: Sequence[Sequence[float]],
    alpha: float,
) -> pd.DataFrame:
    """Evaluate residual normality and variance homogeneity, not independence."""
    rows: List[Dict[str, object]] = []
    residual_values = np.asarray(residuals, dtype=float)
    if 3 <= len(residual_values) <= 5000 and np.ptp(residual_values) > 0:
        normality = stats.shapiro(residual_values)
        rows.append(
            {
                "test": "Shapiro-Wilk (model residuals)",
                "statistic": float(normality.statistic),
                "p_value": float(normality.pvalue),
                "status": "possible deviation" if normality.pvalue < alpha else "no evidence of deviation",
                "note": "A non-significant result does not prove normality.",
            }
        )
    else:
        rows.append(
            {
                "test": "Shapiro-Wilk (model residuals)",
                "statistic": np.nan,
                "p_value": np.nan,
                "status": "not evaluated",
                "note": "Requires 3–5,000 non-constant residuals.",
            }
        )

    if len(groups) >= 2 and all(len(group) >= 2 for group in groups):
        absolute_deviations = [
            np.abs(np.asarray(group, dtype=float) - np.median(group))
            for group in groups
        ]
        if all(np.ptp(deviations) == 0 for deviations in absolute_deviations):
            rows.append(
                {
                    "test": "Levene (median-centered)",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "status": "not evaluated",
                    "note": "The absolute deviations have no within-group spread; "
                    "the test statistic is undefined.",
                }
            )
        else:
            levene = stats.levene(*groups, center="median")
            rows.append(
                {
                    "test": "Levene (median-centered)",
                    "statistic": float(levene.statistic),
                    "p_value": float(levene.pvalue),
                    "status": "possible variance heterogeneity"
                    if levene.pvalue < alpha
                    else "no evidence of variance heterogeneity",
                    "note": "Tests equal variances across the supplied groups.",
                }
            )
    else:
        rows.append(
            {
                "test": "Levene (median-centered)",
                "statistic": np.nan,
                "p_value": np.nan,
                "status": "not evaluated",
                "note": "Each group needs at least two observations.",
            }
        )
    return pd.DataFrame(rows)


def add_effect_sizes(
    table: pd.DataFrame, residual_key: str, total_ss: float = None
) -> pd.DataFrame:
    """Add eta-squared and partial eta-squared where their SS are estimable."""
    result = table.copy()
    residual_ss = float(result.loc[residual_key, "sum_sq"])
    residual_df = float(result.loc[residual_key, "df"])
    mse = residual_ss / residual_df if residual_df > 0 else np.nan
    if total_ss is None:
        total_ss = float(result["sum_sq"].sum())
    eta_squared: List[float] = []
    partial_eta_squared: List[float] = []
    omega_squared: List[float] = []
    for index, row in result.iterrows():
        if index == residual_key:
            eta_squared.append(np.nan)
            partial_eta_squared.append(np.nan)
            omega_squared.append(np.nan)
            continue
        ss = float(row["sum_sq"])
        df = float(row["df"])
        eta_squared.append(ss / total_ss if total_ss > 0 else np.nan)
        partial_denominator = ss + residual_ss
        partial_eta_squared.append(
            ss / partial_denominator if partial_denominator > 0 else np.nan
        )
        omega_squared.append(
            max(0.0, (ss - df * mse) / (total_ss + mse))
            if total_ss + mse > 0
            else np.nan
        )
    result["eta_squared"] = eta_squared
    result["partial_eta_squared"] = partial_eta_squared
    result["omega_squared"] = omega_squared
    return result


def pairwise_tukey(
    groups: Sequence[Tuple[str, Sequence[float]]],
    alpha: float,
    equal_var: bool = True,
) -> pd.DataFrame:
    """Return familywise-adjusted Tukey or Games-Howell comparisons."""
    usable = [(label, np.asarray(values, dtype=float)) for label, values in groups]
    if len(usable) < 2:
        return pd.DataFrame()
    if any(len(values) < 2 for _, values in usable):
        return pd.DataFrame(
            [
                {
                    "group_1": "",
                    "group_2": "",
                    "mean_difference": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_adjusted": np.nan,
                    "reject": False,
                    "note": "Post-hoc comparison requires at least two observations per group.",
                }
            ]
        )
    result = stats.tukey_hsd(
        *(values for _, values in usable), equal_var=equal_var
    )
    intervals = result.confidence_interval(confidence_level=1 - alpha)
    rows = []
    for left in range(len(usable)):
        for right in range(left + 1, len(usable)):
            rows.append(
                {
                    "group_1": usable[left][0],
                    "group_2": usable[right][0],
                    "mean_difference": float(result.statistic[left, right]),
                    "ci_low": float(intervals.low[left, right]),
                    "ci_high": float(intervals.high[left, right]),
                    "p_adjusted": float(result.pvalue[left, right]),
                    "reject": bool(result.pvalue[left, right] < alpha),
                    "note": "Tukey HSD" if equal_var else "Games-Howell",
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Apply Holm's step-down familywise p-value adjustment."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, position in enumerate(order):
        running = max(running, (count - rank) * values[position])
        adjusted[position] = min(1.0, running)
    return adjusted


def pairwise_mann_whitney(
    groups: Sequence[Tuple[str, Sequence[float]]], alpha: float
) -> pd.DataFrame:
    """Provide Holm-adjusted pairwise Mann-Whitney tests after Kruskal-Wallis."""
    usable = [(label, np.asarray(values, dtype=float)) for label, values in groups]
    comparisons: List[Tuple[int, int, float]] = []
    for left in range(len(usable)):
        for right in range(left + 1, len(usable)):
            p_value = stats.mannwhitneyu(
                usable[left][1], usable[right][1], alternative="two-sided"
            ).pvalue
            comparisons.append((left, right, float(p_value)))
    adjusted = holm_adjust([item[2] for item in comparisons])
    return pd.DataFrame(
        [
            {
                "group_1": usable[left][0],
                "group_2": usable[right][0],
                "mean_difference": float(np.median(usable[left][1]) - np.median(usable[right][1])),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_adjusted": float(adjusted[index]),
                "reject": bool(adjusted[index] < alpha),
                "note": "Holm-adjusted pairwise Mann-Whitney; median difference shown; no CI.",
            }
            for index, (left, right, _) in enumerate(comparisons)
        ]
    )
