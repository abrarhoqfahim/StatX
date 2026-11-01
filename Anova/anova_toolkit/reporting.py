"""CSV, Excel, and PDF result writers."""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .data_loader import summarize_dataset
from .results import AnalysisResult


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "response"


def _json_value(value):
    if isinstance(value, dict):
        excluded = {"plot_data", "fitted", "residuals"}
        return {
            str(key): _json_value(item)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _write_excel(results: List[AnalysisResult], path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        used = set()
        for result in results:
            base = _safe_name(result.response)[:20] or "response"
            for suffix, frame in (
                ("anova", result.anova_table),
                ("groups", result.group_summary),
                ("assumptions", result.assumptions),
                ("posthoc", result.posthoc),
            ):
                root = "{}_{}".format(base, suffix)[:31]
                sheet = root
                counter = 2
                while sheet in used:
                    suffix_text = "_{}".format(counter)
                    sheet = root[: 31 - len(suffix_text)] + suffix_text
                    counter += 1
                used.add(sheet)
                frame.to_excel(writer, sheet_name=sheet, index=True)


def _table(data: List[List[object]], widths=None) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183B56")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D2DC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F9")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return "{:.5g}".format(float(value))
    return str(value)


def _write_pdf(
    results: List[AnalysisResult],
    path: Path,
    dataset_summary: Dict[str, object],
    figures: Optional[Dict[str, list]],
) -> None:
    styles = getSampleStyleSheet()
    styles["BodyText"].leading = 13
    story = [
        Paragraph("ANOVA Research Report", styles["Title"]),
        Paragraph("Dataset summary", styles["Heading2"]),
        Paragraph(
            "{} rows, {} columns, {} missing cells, {} duplicate rows.".format(
                dataset_summary["rows"],
                dataset_summary["columns"],
                dataset_summary["missing_cells"],
                dataset_summary["duplicate_rows"],
            ),
            styles["BodyText"],
        ),
        Spacer(1, 0.15 * inch),
    ]
    for index, result in enumerate(results):
        if index:
            story.append(PageBreak())
        story.extend(
            [
                Paragraph(
                    "Response: {}".format(escape(result.response)), styles["Heading1"]
                ),
                Paragraph(
                    "{}; complete observations: {}; excluded for missing selected "
                    "values: {}; alpha: {}.".format(
                        escape(result.model_description),
                        result.complete_n,
                        result.excluded_n,
                        result.alpha,
                    ),
                    styles["BodyText"],
                ),
                Spacer(1, 0.12 * inch),
                Paragraph("ANOVA / test results", styles["Heading2"]),
            ]
        )
        anova = result.anova_table.reset_index()
        anova.columns = [str(column) for column in anova.columns]
        story.append(
            _table(
                [[str(column) for column in anova.columns]]
                + [[_format_cell(value) for value in row] for row in anova.itertuples(index=False, name=None)]
            )
        )
        story.extend([Spacer(1, 0.12 * inch), Paragraph("Group summaries", styles["Heading2"])])
        summary = result.group_summary
        story.append(
            _table(
                [[str(column) for column in summary.columns]]
                + [[_format_cell(value) for value in row] for row in summary.itertuples(index=False, name=None)]
            )
        )
        story.extend([Spacer(1, 0.12 * inch), Paragraph("Assumption checks", styles["Heading2"])])
        assumptions = result.assumptions
        story.append(
            _table(
                [[str(column) for column in assumptions.columns]]
                + [
                    [_format_cell(value) for value in row]
                    for row in assumptions.itertuples(index=False, name=None)
                ]
            )
        )
        if not result.posthoc.empty:
            story.extend([Spacer(1, 0.12 * inch), Paragraph("Post-hoc comparisons", styles["Heading2"])])
            story.append(
                _table(
                    [[str(column) for column in result.posthoc.columns]]
                    + [
                        [_format_cell(value) for value in row]
                        for row in result.posthoc.itertuples(index=False, name=None)
                    ]
                )
            )
        story.extend([Spacer(1, 0.12 * inch), Paragraph("Interpretation and limitations", styles["Heading2"])])
        story.extend(
            [
                Paragraph(escape(item), styles["BodyText"])
                for item in result.interpretation
            ]
        )
        if figures and result.response in figures:
            for figure in figures[result.response]:
                story.extend(
                    [
                        Spacer(1, 0.12 * inch),
                        Image(figure, width=6.5 * inch, height=3.0 * inch, kind="proportional"),
                    ]
                )
    story.extend(
        [
            Spacer(1, 0.2 * inch),
            Paragraph(
                "Inference assumes independent experimental units. Diagnostics do not "
                "prove assumptions; statistical significance does not establish "
                "causality or practical importance.",
                styles["Italic"],
            ),
        ]
    )
    SimpleDocTemplate(
        str(path), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch,
        topMargin=0.55 * inch, bottomMargin=0.55 * inch
    ).build(story)


def write_reports(
    results: Iterable[AnalysisResult],
    output_dir: Path,
    input_path: Path,
    dataset: pd.DataFrame,
    figures: Optional[Dict[str, list]] = None,
) -> Dict[str, str]:
    """Write a reproducible bundle of CSV tables, workbook, and PDF report."""
    result_list = list(results)
    if not result_list:
        raise ValueError("No analysis results were provided for reporting.")
    stems = [_safe_name(result.response) for result in result_list]
    if len(set(stems)) != len(stems):
        raise ValueError("Response names map to duplicate report filenames.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for result in result_list:
        stem = _safe_name(result.response)
        result.anova_table.to_csv(destination / "{}_anova.csv".format(stem), index=True)
        result.group_summary.to_csv(destination / "{}_groups.csv".format(stem), index=False)
        result.assumptions.to_csv(destination / "{}_assumptions.csv".format(stem), index=False)
        result.posthoc.to_csv(destination / "{}_posthoc.csv".format(stem), index=False)
        metadata = {
            "response": result.response,
            "analysis": result.analysis,
            "complete_observations": result.complete_n,
            "excluded_for_missing_selected_values": result.excluded_n,
            "alpha": result.alpha,
            "model": result.model_description,
            "interpretation": result.interpretation,
            "diagnostics": result.diagnostics,
        }
        with (destination / "{}_metadata.json".format(stem)).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(_json_value(metadata), handle, indent=2, ensure_ascii=False)

    workbook_path = destination / "anova_results.xlsx"
    _write_excel(result_list, workbook_path)
    dataset_summary = summarize_dataset(dataset)
    with (destination / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_summary, handle, indent=2, ensure_ascii=False)
    manifest_path = destination / "run_manifest.json"
    manifest = {
        "input_path": str(Path(input_path).expanduser().resolve()),
        "dataset_summary": dataset_summary,
        "responses": [
            {
                "response": result.response,
                "analysis": result.analysis,
                "model": result.model_description,
                "complete_observations": result.complete_n,
                "excluded_for_missing_selected_values": result.excluded_n,
                "alpha": result.alpha,
            }
            for result in result_list
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    pdf_path = destination / "anova_report.pdf"
    _write_pdf(result_list, pdf_path, dataset_summary, figures)
    return {
        "output_directory": str(destination),
        "workbook": str(workbook_path),
        "pdf_report": str(pdf_path),
        "manifest": str(manifest_path),
    }
