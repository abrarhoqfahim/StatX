"""Calculate four correlation coefficients for two numeric CSV columns."""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


Pair = Tuple[float, float]
Result = Tuple[str, Optional[float], str, str]


def read_numeric_pairs(
    csv_path: Path, column_x: str, column_y: str
) -> Tuple[List[Pair], int, int]:
    """Read complete numeric pairs and count total and skipped data rows."""
    pairs = []
    total_rows = 0
    skipped_rows = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames

        if headers is None:
            raise ValueError("The CSV file is empty or has no header row.")

        for column_name in (column_x, column_y):
            if headers.count(column_name) != 1:
                if column_name not in headers:
                    raise ValueError(
                        "Column {!r} was not found. Available columns: {}".format(
                            column_name, ", ".join(headers)
                        )
                    )
                raise ValueError(
                    "Column {!r} appears more than once in the CSV header.".format(
                        column_name
                    )
                )

        if column_x == column_y:
            raise ValueError("Choose two different columns.")

        for record_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    "CSV record {} has a different number of fields than the "
                    "header.".format(record_number)
                )

            total_rows += 1
            raw_x = row[column_x]
            raw_y = row[column_y]
            value_x = _parse_value(raw_x, column_x, record_number)
            value_y = _parse_value(raw_y, column_y, record_number)

            if value_x is None or value_y is None:
                skipped_rows += 1
                continue

            pairs.append((value_x, value_y))

    if not pairs:
        raise ValueError(
            "The selected columns have no complete numeric pairs to analyze."
        )

    return pairs, total_rows, skipped_rows


def _parse_value(
    raw_value: str, column_name: str, record_number: int
) -> Optional[float]:
    if not raw_value.strip():
        return None

    try:
        value = float(raw_value.strip())
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
    return value


def _pearson_r(
    x_values: Sequence[float], y_values: Sequence[float]
) -> Optional[float]:
    if len(x_values) < 2:
        return None

    x_scale = max(abs(value) for value in x_values)
    y_scale = max(abs(value) for value in y_values)
    if x_scale == 0 or y_scale == 0:
        return None

    scaled_x = [value / x_scale for value in x_values]
    scaled_y = [value / y_scale for value in y_values]
    x_mean = math.fsum(scaled_x) / len(scaled_x)
    y_mean = math.fsum(scaled_y) / len(scaled_y)
    x_centered = [value - x_mean for value in scaled_x]
    y_centered = [value - y_mean for value in scaled_y]
    x_sum_squares = math.fsum(value * value for value in x_centered)
    y_sum_squares = math.fsum(value * value for value in y_centered)

    if x_sum_squares == 0 or y_sum_squares == 0:
        return None

    covariance_sum = math.fsum(
        x_value * y_value for x_value, y_value in zip(x_centered, y_centered)
    )
    coefficient = covariance_sum / math.sqrt(x_sum_squares * y_sum_squares)
    return max(-1.0, min(1.0, coefficient))


def _average_ranks(values: Sequence[float]) -> List[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0

    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1

        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            original_index = ordered[position][0]
            ranks[original_index] = average_rank
        start = end

    return ranks


def _tie_pair_count(values: Sequence[float]) -> int:
    counts: Dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sum(count * (count - 1) // 2 for count in counts.values())


def _fenwick_sum(tree: Sequence[int], index: int) -> int:
    total = 0
    while index > 0:
        total += tree[index]
        index -= index & -index
    return total


def _fenwick_add(tree: List[int], index: int) -> None:
    while index < len(tree):
        tree[index] += 1
        index += index & -index


def _kendall_tau_b(pairs: Sequence[Pair]) -> Optional[float]:
    count = len(pairs)
    total_pairs = count * (count - 1) // 2
    ties_x = _tie_pair_count([pair[0] for pair in pairs])
    ties_y = _tie_pair_count([pair[1] for pair in pairs])
    ties_both = _tie_pair_count(pairs)
    denominator = math.sqrt((total_pairs - ties_x) * (total_pairs - ties_y))
    if denominator == 0:
        return None

    ordered_y = sorted(set(pair[1] for pair in pairs))
    y_ranks = {value: index + 1 for index, value in enumerate(ordered_y)}
    ordered_pairs = sorted(pairs)
    tree = [0] * (len(ordered_y) + 1)
    discordant = 0
    previous_count = 0
    group_start = 0

    while group_start < count:
        group_end = group_start + 1
        while (
            group_end < count
            and ordered_pairs[group_end][0] == ordered_pairs[group_start][0]
        ):
            group_end += 1

        group_ranks = [
            y_ranks[ordered_pairs[index][1]]
            for index in range(group_start, group_end)
        ]
        for rank in group_ranks:
            discordant += previous_count - _fenwick_sum(tree, rank)
        for rank in group_ranks:
            _fenwick_add(tree, rank)
            previous_count += 1

        group_start = group_end

    comparable_pairs = total_pairs - ties_x - ties_y + ties_both
    numerator = comparable_pairs - 2 * discordant
    coefficient = numerator / denominator
    return max(-1.0, min(1.0, coefficient))


def calculate_correlations(
    pairs: Sequence[Pair], column_x: str, column_y: str
) -> List[Result]:
    """Calculate Pearson, Spearman, Kendall tau-b, and point-biserial coefficients."""
    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    results: List[Result] = []

    pearson = _pearson_r(x_values, y_values)
    results.append(
        (
            "pearson",
            pearson,
            "ok" if pearson is not None else "undefined",
            "Pearson product-moment correlation.",
        )
    )

    spearman = _pearson_r(_average_ranks(x_values), _average_ranks(y_values))
    results.append(
        (
            "spearman",
            spearman,
            "ok" if spearman is not None else "undefined",
            "Spearman rank correlation using average ranks for ties.",
        )
    )

    kendall = _kendall_tau_b(pairs)
    results.append(
        (
            "kendall_tau_b",
            kendall,
            "ok" if kendall is not None else "undefined",
            "Kendall tau-b, adjusted for ties in either column.",
        )
    )

    unique_x = set(x_values)
    unique_y = set(y_values)
    if len(unique_x) == 2 and len(unique_y) > 2:
        point_biserial = _pearson_r(
            [float(value == max(unique_x)) for value in x_values], y_values
        )
        binary_column = column_x
    elif len(unique_y) == 2 and len(unique_x) > 2:
        point_biserial = _pearson_r(
            [float(value == max(unique_y)) for value in y_values], x_values
        )
        binary_column = column_y
    else:
        point_biserial = None
        binary_column = ""

    if point_biserial is not None:
        point_status = "ok"
        point_note = (
            "Point-biserial correlation; {!r} coded 0 for its lower observed "
            "value and 1 for its higher observed value.".format(binary_column)
        )
    elif binary_column:
        point_status = "undefined"
        point_note = (
            "Point-biserial correlation is undefined because one variable has "
            "zero variance."
        )
    else:
        point_status = "not_applicable"
        point_note = (
            "Requires exactly one binary column and one column with more than "
            "two distinct observed values (treated as continuous)."
        )

    results.append(
        ("point_biserial", point_biserial, point_status, point_note)
    )
    return results


def write_results(
    output_path: Path,
    results: Sequence[Result],
    column_x: str,
    column_y: str,
    complete_pairs: int,
    total_rows: int,
    skipped_rows: int,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "method",
                "column_x",
                "column_y",
                "coefficient",
                "n",
                "total_rows",
                "skipped_rows",
                "status",
                "note",
            ]
        )
        for method, coefficient, status, note in results:
            writer.writerow(
                [
                    method,
                    column_x,
                    column_y,
                    "" if coefficient is None else "{:.12g}".format(coefficient),
                    complete_pairs,
                    total_rows,
                    skipped_rows,
                    status,
                    note,
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Pearson, Spearman, Kendall tau-b, and point-biserial "
            "correlations for two numeric CSV columns."
        )
    )
    parser.add_argument("csv_file", type=Path, help="Path to the input CSV file")
    parser.add_argument("column_x", help="Exact name of the first numeric column")
    parser.add_argument("column_y", help="Exact name of the second numeric column")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Path for the results CSV (default: <input name>_correlations.csv "
            "next to the input)"
        ),
    )
    args = parser.parse_args()

    output_path = args.output or args.csv_file.with_name(
        "{}_correlations.csv".format(args.csv_file.stem)
    )
    try:
        if args.csv_file.resolve() == output_path.resolve():
            raise ValueError("The output file must not overwrite the input CSV.")

        pairs, total_rows, skipped_rows = read_numeric_pairs(
            args.csv_file, args.column_x, args.column_y
        )
        results = calculate_correlations(pairs, args.column_x, args.column_y)
        write_results(
            output_path,
            results,
            args.column_x,
            args.column_y,
            len(pairs),
            total_rows,
            skipped_rows,
        )
    except (OSError, csv.Error, ValueError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1

    print("Complete pairs: {}".format(len(pairs)))
    print("Rows skipped for blank values: {}".format(skipped_rows))
    print("Results saved to: {}".format(output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
