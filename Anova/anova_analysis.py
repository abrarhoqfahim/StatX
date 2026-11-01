"""Command-line and interactive entry point for ANOVA analysis."""

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

from anova_toolkit.config import OneWayConfig, TwoWayConfig
from anova_toolkit.data_loader import load_csv, summarize_dataset
from anova_toolkit.one_way_anova import analyze_one_way
from anova_toolkit.reporting import write_reports
from anova_toolkit.two_way_anova import analyze_two_way
from anova_toolkit.visualizations import generate_plots


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run research-oriented one-way or two-way ANOVA on CSV data."
    )
    parser.add_argument("--input", type=Path, help="Path to a CSV input file.")
    parser.add_argument(
        "--responses",
        nargs="+",
        help="One or more numeric response column names, exactly as they appear in the CSV.",
    )
    parser.add_argument("--factor", help="First categorical grouping column.")
    parser.add_argument("--factor-b", help="Second factor (required for two-way ANOVA).")
    parser.add_argument(
        "--analysis",
        choices=("one-way", "two-way"),
        help="Analysis type. Defaults to one-way when --factor-b is omitted.",
    )
    parser.add_argument(
        "--method",
        choices=("classical", "welch", "kruskal"),
        default="classical",
        help="One-way method (default: classical).",
    )
    parser.add_argument(
        "--sum-of-squares",
        choices=("I", "II", "III"),
        default="II",
        help="Two-way sums of squares; sum contrasts are used (default: II).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("anova_results"),
        help="Directory for CSV, Excel, PDF, JSON, and figure outputs.",
    )
    parser.add_argument(
        "--confirm-independent-observations",
        action="store_true",
        help="Confirm each row is an independent experimental unit.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for omitted analysis inputs.",
    )
    return parser


def _prompt_for_missing(args: argparse.Namespace) -> None:
    if args.input is None:
        args.input = Path(input("Input CSV path: ").strip()).expanduser()
    if not args.responses:
        args.responses = [
            item.strip()
            for item in input("Response column(s), comma-separated: ").split(",")
            if item.strip()
        ]
    if args.factor is None:
        args.factor = input("First factor column: ").strip()
    if args.analysis is None and args.factor_b is None:
        args.analysis = input("Analysis type (one-way/two-way) [one-way]: ").strip() or "one-way"
    if args.analysis == "two-way" and args.factor_b is None:
        args.factor_b = input("Second factor column: ").strip()
    if not args.confirm_independent_observations:
        answer = input(
            "Are rows independent experimental units? Enter yes to confirm: "
        ).strip().lower()
        args.confirm_independent_observations = answer in {"yes", "y"}


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.interactive:
        _prompt_for_missing(args)
    if args.input is None:
        parser.error("--input is required unless --interactive is used.")
    if not args.responses:
        parser.error("--responses requires at least one column.")
    if not args.factor:
        parser.error("--factor is required.")
    if not args.confirm_independent_observations:
        parser.error(
            "Confirm row independence with --confirm-independent-observations; "
            "repeated measures need a different model."
        )
    analysis = args.analysis or ("two-way" if args.factor_b else "one-way")
    if analysis == "two-way" and not args.factor_b:
        parser.error("--factor-b is required for two-way ANOVA.")
    if analysis == "one-way" and args.factor_b:
        parser.error("--factor-b can only be used with two-way ANOVA.")

    input_path = args.input.expanduser()
    try:
        frame = load_csv(input_path)
        print("Dataset summary: {}".format(summarize_dataset(frame)))
        if analysis == "one-way":
            config = OneWayConfig(
                input_path=input_path,
                responses=tuple(args.responses),
                factor_a=args.factor,
                alpha=args.alpha,
                independent_observations_confirmed=True,
                method=args.method,
            )
            results = analyze_one_way(config)
        else:
            config = TwoWayConfig(
                input_path=input_path,
                responses=tuple(args.responses),
                factor_a=args.factor,
                factor_b=args.factor_b,
                alpha=args.alpha,
                sum_of_squares=args.sum_of_squares,
                independent_observations_confirmed=True,
            )
            results = analyze_two_way(config)

        result_list = list(results.values())
        figures = generate_plots(result_list, args.output_dir)
        outputs = write_reports(result_list, args.output_dir, input_path, frame, figures)
    except (OSError, ValueError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    for result in result_list:
        print(
            "{}: {} complete observations, {} excluded.".format(
                result.response, result.complete_n, result.excluded_n
            )
        )
        for line in result.interpretation:
            print("  {}".format(line))
    print("Results saved to {}".format(outputs["output_directory"]))
    print("Excel workbook: {}".format(outputs["workbook"]))
    print("PDF report: {}".format(outputs["pdf_report"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
