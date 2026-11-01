"""CSV input and initial dataset summary."""

import csv
from pathlib import Path
from typing import Dict, Union

import pandas as pd


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a UTF-8 CSV without silently converting malformed rows."""
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError("Input CSV does not exist: {}".format(csv_path))
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            headers = next(csv.reader(csv_file), None)
        if not headers:
            raise ValueError("Input CSV is empty or has no header row.")
        if len(headers) != len(set(headers)):
            raise ValueError("Input CSV contains duplicated column headers.")
        frame = pd.read_csv(
            csv_path, encoding="utf-8-sig", keep_default_na=False, na_values=[""]
        )
    except ValueError:
        raise
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError("Could not read CSV {!s}: {}".format(csv_path, error)) from error
    if frame.empty:
        raise ValueError("Input CSV contains no data rows.")
    if len(frame.columns) == 0:
        raise ValueError("Input CSV has no header row.")
    return frame


def summarize_dataset(frame: pd.DataFrame) -> Dict[str, object]:
    """Return a compact structural summary without modifying the data."""
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": [str(name) for name in frame.columns],
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "dtypes": {str(name): str(dtype) for name, dtype in frame.dtypes.items()},
    }
