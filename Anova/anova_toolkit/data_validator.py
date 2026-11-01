"""Validation for response and grouping columns."""

from typing import Iterable

import numpy as np
import pandas as pd


def validate_dataset(
    frame: pd.DataFrame, responses: Iterable[str], factors: Iterable[str]
) -> None:
    """Reject ambiguous columns and invalid selected response data."""
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError("CSV contains duplicated column headers: {!r}".format(duplicates))

    response_names = tuple(responses)
    factor_names = tuple(factors)
    selected = response_names + factor_names
    missing = [name for name in selected if name not in frame.columns]
    if missing:
        raise ValueError(
            "Selected column(s) not found: {}. Available columns: {}".format(
                ", ".join(missing), ", ".join(str(name) for name in frame.columns)
            )
        )
    if set(response_names) & set(factor_names):
        raise ValueError("Response and factor columns must be different.")

    for name in response_names:
        try:
            source = frame[name].replace(r"^\s*$", np.nan, regex=True)
            values = pd.to_numeric(source, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Response column {!r} must contain numeric values; "
                "invalid values are not automatically discarded.".format(name)
            ) from error
        nonmissing = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(nonmissing).all():
            raise ValueError(
                "Response column {!r} contains infinite values.".format(name)
            )

    for name in factor_names:
        observed = frame[name].dropna()
        if observed.empty:
            raise ValueError("Factor column {!r} contains no observed values.".format(name))
        if pd.api.types.is_numeric_dtype(observed.dtype) and not np.isfinite(
            observed.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Numeric factor column {!r} contains infinite values.".format(name)
            )
        if observed.nunique() < 2:
            raise ValueError("Factor column {!r} must have at least two levels.".format(name))
