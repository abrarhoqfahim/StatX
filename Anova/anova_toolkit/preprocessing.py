"""Explicit complete-case preparation and group summaries."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def prepare_response(
    frame: pd.DataFrame, response: str, factors: Tuple[str, ...]
) -> Tuple[pd.DataFrame, int]:
    """Return selected complete cases and the number of excluded data rows."""
    selected = frame.loc[:, [response, *factors]].copy()
    selected = selected.replace(r"^\s*$", np.nan, regex=True)
    selected[response] = pd.to_numeric(selected[response], errors="raise")
    selected = selected.replace([np.inf, -np.inf], np.nan)
    for factor in factors:
        levels = selected[factor].dropna().unique().tolist()
        selected[factor] = pd.Categorical(selected[factor], categories=levels)
    complete = selected.dropna(subset=[response, *factors]).copy()
    excluded = len(selected) - len(complete)
    if complete.empty:
        raise ValueError(
            "No complete observations remain for response {!r}.".format(response)
        )
    return complete, excluded


def group_descriptives(
    frame: pd.DataFrame,
    response: str,
    factors: Tuple[str, ...],
    alpha: float,
) -> pd.DataFrame:
    """Summarize groups and report t-based confidence intervals for means."""
    rows: List[Dict[str, object]] = []
    grouped = frame.groupby(list(factors), observed=True, sort=True, dropna=False)
    for keys, values in grouped[response]:
        if not isinstance(keys, tuple):
            keys = (keys,)
        numeric = values.to_numpy(dtype=float)
        n = len(numeric)
        mean = float(np.mean(numeric))
        standard_deviation = float(np.std(numeric, ddof=1)) if n > 1 else float("nan")
        if n > 1:
            half_width = float(
                stats.t.ppf(1 - alpha / 2, n - 1) * standard_deviation / np.sqrt(n)
            )
            ci_low, ci_high = mean - half_width, mean + half_width
        else:
            ci_low, ci_high = float("nan"), float("nan")
        row: Dict[str, object] = {
            factor: key for factor, key in zip(factors, keys)
        }
        row.update(
            {
                "n": n,
                "mean": mean,
                "standard_deviation": standard_deviation,
                "standard_error": standard_deviation / np.sqrt(n)
                if n > 1
                else float("nan"),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
