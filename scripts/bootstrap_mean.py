#!/usr/bin/env python3
"""Bootstrap confidence interval for a sample mean from a CSV column."""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional


def read_numeric_column(csv_path: Path, column_name: str) -> List[float]:
    """Read a numeric column from a CSV file while skipping blanks."""
    values: List[float] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("The CSV file is empty or has no header row.")

        if column_name not in reader.fieldnames:
            available = ", ".join(reader.fieldnames) if reader.fieldnames else "(none)"
            raise ValueError(
                "Column {!r} was not found. Available columns: {}".format(
                    column_name, available
                )
            )
        if reader.fieldnames.count(column_name) != 1:
            raise ValueError(
                "Column {!r} appears more than once in the CSV header.".format(
                    column_name
                )
            )

        for record_number, row in enumerate(reader, start=2):
            if row is None:
                continue
            if None in row:
                raise ValueError(
                    "CSV record {} has more fields than the header.".format(
                        record_number
                    )
                )
            raw_value = row.get(column_name)
            if raw_value is None or not str(raw_value).strip():
                continue
            try:
                value = float(raw_value)
            except ValueError as error:
                raise ValueError(
                    "Non-numeric value {!r} in column {!r} at CSV record {}.".format(
                        raw_value, column_name, record_number
                    )
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    "Non-finite value {!r} in column {!r} at CSV record {}.".format(
                        raw_value, column_name, record_number
                    )
                )
            values.append(value)

    if not values:
        raise ValueError("Column {!r} contains no numeric values.".format(column_name))
    return values


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("No values available for percentiles.")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    if q <= 0:
        return float(ordered[0])
    if q >= 100:
        return float(ordered[-1])
    index = (len(ordered) - 1) * (q / 100.0)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(ordered[lower])
    fraction = index - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def bootstrap_mean_summary(
    sample: List[float],
    n_resamples: int,
    confidence_level: float,
    seed: Optional[int],
) -> Dict[str, object]:
    if n_resamples <= 0:
        raise ValueError("Number of bootstrap resamples must be positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    sample_mean = sum(sample) / len(sample)
    rng = random.Random(seed)
    bootstrap_means: List[float] = []

    for _ in range(n_resamples):
        resample = [sample[rng.randrange(len(sample))] for _ in sample]
        bootstrap_means.append(sum(resample) / len(resample))

    bootstrap_mean = sum(bootstrap_means) / len(bootstrap_means)
    if len(bootstrap_means) > 1:
        variance = sum((value - bootstrap_mean) ** 2 for value in bootstrap_means) / (
            len(bootstrap_means) - 1
        )
        bootstrap_se = math.sqrt(variance)
    else:
        bootstrap_se = 0.0

    alpha = 1.0 - confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = percentile(bootstrap_means, lower_q)
    ci_upper = percentile(bootstrap_means, upper_q)

    return {
        "sample_mean": float(sample_mean),
        "bootstrap_mean": float(bootstrap_mean),
        "bootstrap_se": float(bootstrap_se),
        "confidence_level": float(confidence_level),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "bootstrap_means": [float(value) for value in bootstrap_means],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the mean of a numeric CSV column with nonparametric bootstrap "
            "confidence intervals."
        )
    )
    parser.add_argument("csv_file", type=Path, help="Path to the CSV file containing the data.")
    parser.add_argument("column", help="Exact name of the numeric column to analyze.")
    parser.add_argument(
        "--n",
        type=int,
        default=10000,
        help="Number of bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for the bootstrap interval (default: 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible bootstrap resampling (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the JSON result summary.",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()

    try:
        data = read_numeric_column(args.csv_file, args.column)
        summary = bootstrap_mean_summary(
            data,
            n_resamples=args.n,
            confidence_level=args.confidence_level,
            seed=args.seed,
        )
    except (OSError, ValueError, csv.Error) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1

    payload = json.dumps(summary, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
