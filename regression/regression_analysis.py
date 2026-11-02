"""Fit linear, multiple linear, or polynomial OLS models to CSV data."""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import statsmodels.api as sm

if __package__:
    from .diagnostic_plots import save_diagnostic_plots
else:
    from diagnostic_plots import save_diagnostic_plots


def read_regression_data(
    csv_path: Path, target: str, predictors: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Read complete numeric cases for the target and predictor columns."""
    selected_columns = [target] + list(predictors)
    predictor_rows: List[List[float]] = []
    target_values: List[float] = []
    total_rows = 0
    excluded_rows = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError("The CSV file is empty or has no header row.")
        if len(headers) != len(set(headers)):
            raise ValueError("The CSV file contains duplicated column headers.")
        if any(not header.strip() for header in headers):
            raise ValueError("The CSV file contains an empty column header.")

        for column_name in selected_columns:
            if column_name not in headers:
                raise ValueError(
                    "Column {!r} was not found. Available columns: {}".format(
                        column_name, ", ".join(headers)
                    )
                )

        for record_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    "CSV record {} has a different number of fields than the "
                    "header.".format(record_number)
                )
            total_rows += 1
            raw_values = [row[column_name] for column_name in selected_columns]
            numeric_values: List[float] = []
            has_missing_value = False
            for column_name, raw_value in zip(selected_columns, raw_values):
                if not raw_value.strip():
                    has_missing_value = True
                    continue
                try:
                    value = float(raw_value.strip())
                except ValueError as error:
                    raise ValueError(
                        "Non-numeric value {!r} in column {!r} at CSV record "
                        "{}.".format(raw_value, column_name, record_number)
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        "Non-finite value {!r} in column {!r} at CSV record "
                        "{}.".format(raw_value, column_name, record_number)
                    )
                numeric_values.append(value)

            if has_missing_value:
                excluded_rows += 1
                continue
            target_values.append(numeric_values[0])
            predictor_rows.append(numeric_values[1:])

    if not target_values:
        raise ValueError("No complete numeric rows are available for regression.")

    return (
        np.asarray(predictor_rows, dtype=float),
        np.asarray(target_values, dtype=float),
        total_rows,
        excluded_rows,
    )


def create_design_matrix(
    predictor_values: np.ndarray,
    predictor_names: Sequence[str],
    model_type: str,
    degree: Optional[int],
) -> Tuple[np.ndarray, List[str], Optional[str]]:
    """Build the intercept and predictor terms for the selected model."""
    if model_type == "linear":
        if len(predictor_names) != 1:
            raise ValueError("Linear regression requires exactly one predictor.")
        design = np.column_stack((np.ones(len(predictor_values)), predictor_values))
        return design, ["Intercept", predictor_names[0]], None

    if model_type == "multiple":
        if len(predictor_names) < 2:
            raise ValueError(
                "Multiple linear regression requires at least two predictors."
            )
        design = np.column_stack((np.ones(len(predictor_values)), predictor_values))
        return design, ["Intercept"] + list(predictor_names), None

    if model_type == "polynomial":
        if len(predictor_names) != 1:
            raise ValueError("Polynomial regression requires exactly one predictor.")
        if degree is None or degree < 2:
            raise ValueError("Polynomial regression degree must be at least 2.")

        predictor = predictor_values[:, 0]
        center = float(np.mean(predictor))
        scale = float(np.std(predictor))
        if not math.isfinite(scale) or scale == 0:
            raise ValueError(
                "Polynomial regression requires a predictor with nonzero variation."
            )
        standardized = (predictor - center) / scale
        columns = [np.ones(len(predictor))]
        columns.extend(standardized ** power for power in range(1, degree + 1))
        names = ["Intercept"] + [
            "z^{}".format(power) for power in range(1, degree + 1)
        ]
        basis = "z = ({} - {:.10g}) / {:.10g}".format(
            predictor_names[0], center, scale
        )
        return np.column_stack(columns), names, basis

    raise ValueError("Unsupported regression model {!r}.".format(model_type))


def _format_stat(value: Any) -> str:
    """Format a finite statistic, making undefined values explicit."""
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return "undefined"
    return "{:.10g}".format(numeric_value)


def _format_equation(
    params: np.ndarray, coefficient_names: Sequence[str]
) -> str:
    equation = "({})".format(_format_stat(params[0]))
    equation_terms = [
        "({})*{}".format(_format_stat(coefficient), name)
        for coefficient, name in zip(params[1:], coefficient_names[1:])
    ]
    if equation_terms:
        equation += " + " + " + ".join(equation_terms)
    return "y_hat = {}".format(equation)


def format_report(
    fitted_model: Any,
    coefficient_names: Sequence[str],
    model_type: str,
    target: str,
    predictor_names: Sequence[str],
    basis: Optional[str],
    total_rows: int,
    excluded_rows: int,
    confidence_level: float,
) -> str:
    """Format model fit statistics and coefficient inference for the CLI."""
    params = np.asarray(fitted_model.params, dtype=float)
    standard_errors = np.asarray(fitted_model.bse, dtype=float)
    t_values = np.asarray(fitted_model.tvalues, dtype=float)
    p_values = np.asarray(fitted_model.pvalues, dtype=float)
    intervals = np.asarray(
        fitted_model.conf_int(alpha=1.0 - confidence_level), dtype=float
    )
    equation = _format_equation(params, coefficient_names)
    rmse = math.sqrt(
        float(fitted_model.mse_resid)
        * float(fitted_model.df_resid)
        / float(fitted_model.nobs)
    )

    lines = [
        "Regression analysis",
        "Model: {}".format(model_type),
        "Target: {}".format(target),
        "Predictors: {}".format(", ".join(predictor_names)),
        "Complete observations: {}".format(int(fitted_model.nobs)),
        "Rows excluded for missing selected values: {}".format(excluded_rows),
        "Rows read: {}".format(total_rows),
        "Equation: {}".format(equation),
    ]
    if basis is not None:
        lines.append("Polynomial basis: {}".format(basis))

    lines.extend(
        [
            "R-squared: {}".format(_format_stat(fitted_model.rsquared)),
            "Adjusted R-squared: {}".format(
                _format_stat(fitted_model.rsquared_adj)
            ),
            "F-statistic: {}".format(_format_stat(fitted_model.fvalue)),
            "Model F-test p-value: {}".format(_format_stat(fitted_model.f_pvalue)),
            "Model degrees of freedom: {}".format(
                _format_stat(fitted_model.df_model)
            ),
            "Residual degrees of freedom: {}".format(
                _format_stat(fitted_model.df_resid)
            ),
            "RMSE (in-sample): {}".format(_format_stat(rmse)),
            "Residual standard error: {}".format(
                _format_stat(math.sqrt(float(fitted_model.mse_resid)))
            ),
            "AIC: {}".format(_format_stat(fitted_model.aic)),
            "BIC: {}".format(_format_stat(fitted_model.bic)),
            "Coefficient inference ({}% confidence intervals; "
            "two-sided t tests):".format(_format_stat(100.0 * confidence_level)),
            "Term\tEstimate\tStd. error\tt\tp-value\tCI lower\tCI upper",
        ]
    )

    for index, name in enumerate(coefficient_names):
        lines.append(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                name,
                _format_stat(params[index]),
                _format_stat(standard_errors[index]),
                _format_stat(t_values[index]),
                _format_stat(p_values[index]),
                _format_stat(intervals[index, 0]),
                _format_stat(intervals[index, 1]),
            )
        )
    return "\n".join(lines)


def write_report_csv(
    output_path: Path,
    fitted_model: Any,
    coefficient_names: Sequence[str],
    model_type: str,
    target: str,
    predictor_names: Sequence[str],
    basis: Optional[str],
    total_rows: int,
    excluded_rows: int,
    confidence_level: float,
) -> None:
    """Write model metadata, fit statistics, and coefficient inference to CSV."""
    params = np.asarray(fitted_model.params, dtype=float)
    standard_errors = np.asarray(fitted_model.bse, dtype=float)
    t_values = np.asarray(fitted_model.tvalues, dtype=float)
    p_values = np.asarray(fitted_model.pvalues, dtype=float)
    intervals = np.asarray(
        fitted_model.conf_int(alpha=1.0 - confidence_level), dtype=float
    )
    rmse = math.sqrt(
        float(fitted_model.mse_resid)
        * float(fitted_model.df_resid)
        / float(fitted_model.nobs)
    )
    rows = [
        {"section": "model", "metric": "model type", "value": model_type},
        {"section": "model", "metric": "target", "value": target},
        {
            "section": "model",
            "metric": "predictors",
            "value": ", ".join(predictor_names),
        },
        {
            "section": "model",
            "metric": "complete observations",
            "value": str(int(fitted_model.nobs)),
        },
        {
            "section": "model",
            "metric": "rows excluded for missing selected values",
            "value": str(excluded_rows),
        },
        {"section": "model", "metric": "rows read", "value": str(total_rows)},
        {
            "section": "model",
            "metric": "confidence level",
            "value": _format_stat(confidence_level),
        },
        {
            "section": "model",
            "metric": "fitted equation",
            "value": _format_equation(params, coefficient_names),
        },
    ]
    if basis is not None:
        rows.append(
            {"section": "model", "metric": "polynomial basis", "value": basis}
        )

    rows.extend(
        [
            {
                "section": "fit",
                "metric": "R-squared",
                "value": _format_stat(fitted_model.rsquared),
            },
            {
                "section": "fit",
                "metric": "adjusted R-squared",
                "value": _format_stat(fitted_model.rsquared_adj),
            },
            {
                "section": "fit",
                "metric": "model F-test",
                "value": _format_stat(fitted_model.fvalue),
                "statistic": _format_stat(fitted_model.fvalue),
                "p_value": _format_stat(fitted_model.f_pvalue),
            },
            {
                "section": "fit",
                "metric": "model degrees of freedom",
                "value": _format_stat(fitted_model.df_model),
            },
            {
                "section": "fit",
                "metric": "residual degrees of freedom",
                "value": _format_stat(fitted_model.df_resid),
            },
            {
                "section": "fit",
                "metric": "RMSE (in-sample)",
                "value": _format_stat(rmse),
            },
            {
                "section": "fit",
                "metric": "residual standard error",
                "value": _format_stat(math.sqrt(float(fitted_model.mse_resid))),
            },
            {"section": "fit", "metric": "AIC", "value": _format_stat(fitted_model.aic)},
            {"section": "fit", "metric": "BIC", "value": _format_stat(fitted_model.bic)},
        ]
    )

    for index, name in enumerate(coefficient_names):
        rows.append(
            {
                "section": "coefficient",
                "metric": name,
                "value": _format_stat(params[index]),
                "standard_error": _format_stat(standard_errors[index]),
                "statistic": _format_stat(t_values[index]),
                "p_value": _format_stat(p_values[index]),
                "ci_lower": _format_stat(intervals[index, 0]),
                "ci_upper": _format_stat(intervals[index, 1]),
            }
        )

    fieldnames = [
        "section",
        "metric",
        "value",
        "standard_error",
        "statistic",
        "p_value",
        "ci_lower",
        "ci_upper",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a linear, multiple linear, or polynomial OLS regression to CSV data."
        )
    )
    parser.add_argument("csv_file", type=Path, help="Path to the input CSV file.")
    parser.add_argument("--target", required=True, help="Numeric response column.")
    parser.add_argument(
        "--predictors",
        nargs="+",
        required=True,
        help="One or more numeric predictor columns.",
    )
    parser.add_argument(
        "--model",
        choices=("linear", "multiple", "polynomial"),
        required=True,
        help="Regression model to fit.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        help="Polynomial degree (at least 2; default 2 for polynomial models).",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Coefficient confidence level between 0 and 1 (default: 0.95).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Path for the results CSV (default: <input-name>_regression_results.csv "
            "beside the input file)."
        ),
    )
    return parser


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Run the regression command and return its process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    degree = args.degree

    if not 0.0 < args.confidence_level < 1.0:
        parser.error("--confidence-level must be strictly between 0 and 1.")
    if len(args.predictors) != len(set(args.predictors)):
        parser.error("--predictors must not contain duplicate column names.")
    if args.target in args.predictors:
        parser.error("The target column cannot also be a predictor.")
    if args.model == "polynomial":
        degree = 2 if degree is None else degree
        if degree < 2:
            parser.error("--degree must be at least 2 for polynomial regression.")
    elif degree is not None:
        parser.error("--degree can only be used with polynomial regression.")

    try:
        input_path = args.csv_file.expanduser().resolve()
        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else input_path.with_name(
                "{}_regression_results.csv".format(input_path.stem)
            )
        )
        if output_path == input_path:
            raise ValueError("The results CSV cannot overwrite the input CSV.")
        data_directory = Path(__file__).resolve().parents[1] / "data"
        plot_path = data_directory / "{}_{}_diagnostics.png".format(
            input_path.stem, args.model
        )
        predictor_values, target_values, total_rows, excluded_rows = (
            read_regression_data(
                input_path, args.target, args.predictors
            )
        )
        design, coefficient_names, basis = create_design_matrix(
            predictor_values, args.predictors, args.model, degree
        )
        if not np.isfinite(design).all():
            raise ValueError(
                "The model design contains non-finite terms; check predictor "
                "magnitude and polynomial degree."
            )
        if design.shape[0] <= design.shape[1]:
            raise ValueError(
                "The model needs more complete observations than fitted "
                "coefficients."
            )
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError(
                "The model design is rank deficient; predictors or polynomial "
                "terms are linearly dependent."
            )

        fitted_model = sm.OLS(target_values, design).fit()
        save_diagnostic_plots(
            output_path=plot_path,
            fitted_model=fitted_model,
            predictor_values=predictor_values,
            target_values=target_values,
            predictor_names=args.predictors,
            model_type=args.model,
        )
        report = format_report(
            fitted_model=fitted_model,
            coefficient_names=coefficient_names,
            model_type=args.model,
            target=args.target,
            predictor_names=args.predictors,
            basis=basis,
            total_rows=total_rows,
            excluded_rows=excluded_rows,
            confidence_level=args.confidence_level,
        )
        write_report_csv(
            output_path=output_path,
            fitted_model=fitted_model,
            coefficient_names=coefficient_names,
            model_type=args.model,
            target=args.target,
            predictor_names=args.predictors,
            basis=basis,
            total_rows=total_rows,
            excluded_rows=excluded_rows,
            confidence_level=args.confidence_level,
        )
    except (OSError, UnicodeError, csv.Error, ValueError, np.linalg.LinAlgError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1

    print(report)
    print("Results CSV saved to {}".format(output_path))
    print("Diagnostic plots saved to {}".format(plot_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
