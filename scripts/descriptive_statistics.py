"""Calculate descriptive statistics for a numeric column in a CSV file."""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import List, Tuple


def read_numeric_column(csv_path: Path, column_name: str) -> Tuple[List[float], int]:
    """Read numeric values from a CSV column and count blank cells."""
    values = []
    missing_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames

        if headers is None:
            raise ValueError("The CSV file is empty or has no header row.")
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

        for record_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    "CSV record {} has more fields than the header.".format(
                        record_number
                    )
                )

            raw_value = row.get(column_name)
            if raw_value is None or not raw_value.strip():
                missing_count += 1
                continue

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
            values.append(value)

    if not values:
        raise ValueError(
            "Column {!r} contains no numeric values.".format(column_name)
        )

    return values, missing_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the mean, population variance and standard deviation, "
            "and sample variance and standard deviation for a CSV column."
        )
    )
    parser.add_argument("csv_file", type=Path, help="Path to the input CSV file")
    parser.add_argument("column", help="Exact name of the numeric CSV column")
    args = parser.parse_args()

    try:
        values, missing_count = read_numeric_column(args.csv_file, args.column)
    except (OSError, csv.Error, ValueError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1

    count = len(values)
    print("File: {}".format(args.csv_file))
    print("Column: {}".format(args.column))
    print("Numeric values: {}".format(count))
    print("Blank values skipped: {}".format(missing_count))
    print("Mean: {:.10g}".format(statistics.mean(values)))
    print("Population variance (divide by n): {:.10g}".format(
        statistics.pvariance(values)
    ))
    print("Population standard deviation: {:.10g}".format(statistics.pstdev(values)))

    if count > 1:
        print("Sample variance (divide by n - 1): {:.10g}".format(
            statistics.variance(values)
        ))
        print("Sample standard deviation: {:.10g}".format(statistics.stdev(values)))
    else:
        print("Sample variance: undefined (at least 2 numeric values are required)")
        print(
            "Sample standard deviation: undefined "
            "(at least 2 numeric values are required)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
