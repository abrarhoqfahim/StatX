"""Scientific and workflow tests for the ANOVA toolkit."""

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from scipy import stats

from Anova.anova_toolkit.config import OneWayConfig, TwoWayConfig
from Anova.anova_toolkit.data_loader import load_csv
from Anova.anova_toolkit.one_way_anova import analyze_one_way
from Anova.anova_toolkit.reporting import write_reports
from Anova.anova_toolkit.two_way_anova import analyze_two_way
from Anova.anova_toolkit.visualizations import generate_plots


class AnovaToolkitTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.one_way_path = self.root / "one_way.csv"
        self.two_way_path = self.root / "two_way.csv"
        self._write_one_way()
        self._write_two_way()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_one_way(self):
        with self.one_way_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["outcome", "group"])
            for group, values in (
                ("low", [1, 2, 3, 4]),
                ("middle", [3, 4, 5, 6]),
                ("high", [8, 9, 10, 11]),
            ):
                writer.writerows((value, group) for value in values)

    def _write_two_way(self):
        means = {
            ("A0", "B0"): 5,
            ("A0", "B1"): 7,
            ("A1", "B0"): 8,
            ("A1", "B1"): 15,
        }
        with self.two_way_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["outcome", "factor_a", "factor_b"])
            for (factor_a, factor_b), mean in means.items():
                writer.writerows(
                    (mean + offset, factor_a, factor_b)
                    for offset in (-1.5, -0.5, 0.5, 1.5)
                )

    def one_way_config(self, **overrides):
        values = {
            "input_path": self.one_way_path,
            "responses": ("outcome",),
            "factor_a": "group",
            "independent_observations_confirmed": True,
        }
        values.update(overrides)
        return OneWayConfig(**values)

    def test_one_way_classical_detects_known_group_effect(self):
        result = analyze_one_way(self.one_way_config())["outcome"]
        self.assertEqual(result.complete_n, 12)
        self.assertEqual(result.excluded_n, 0)
        self.assertAlmostEqual(
            result.group_summary.set_index("group").loc["low", "mean"], 2.5
        )
        self.assertLess(result.anova_table.loc["factor", "PR(>F)"], 0.05)
        self.assertIn("eta_squared", result.anova_table.columns)
        self.assertEqual(len(result.posthoc), 3)
        expected = stats.f_oneway([1, 2, 3, 4], [3, 4, 5, 6], [8, 9, 10, 11])
        self.assertAlmostEqual(result.anova_table.loc["factor", "F"], expected.statistic)
        self.assertTrue((result.posthoc["ci_low"] < result.posthoc["ci_high"]).all())

    def test_independence_confirmation_is_required(self):
        with self.assertRaisesRegex(ValueError, "Confirm that rows are independent"):
            analyze_one_way(
                self.one_way_config(independent_observations_confirmed=False)
            )

    def test_welch_and_kruskal_alternatives_are_available(self):
        welch = analyze_one_way(self.one_way_config(method="welch"))["outcome"]
        kruskal = analyze_one_way(self.one_way_config(method="kruskal"))["outcome"]
        self.assertIn("Denominator df", welch.anova_table.index)
        self.assertTrue(welch.posthoc["note"].str.contains("Games-Howell").all())
        expected_welch = stats.f_oneway(
            [1, 2, 3, 4], [3, 4, 5, 6], [8, 9, 10, 11], equal_var=False
        )
        self.assertAlmostEqual(
            welch.anova_table.loc["factor", "F"], expected_welch.statistic
        )
        self.assertIn("H", kruskal.anova_table.columns)
        self.assertTrue(kruskal.posthoc["note"].str.contains("Holm-adjusted").all())

    def test_multiple_responses_are_analyzed_independently(self):
        frame = pd.read_csv(self.one_way_path)
        frame["outcome_2"] = frame["outcome"] * 2
        frame.to_csv(self.one_way_path, index=False)
        result = analyze_one_way(
            self.one_way_config(responses=("outcome", "outcome_2"))
        )
        self.assertEqual(set(result), {"outcome", "outcome_2"})
        self.assertEqual(result["outcome_2"].complete_n, 12)

    def test_missing_selected_values_are_reported_as_excluded(self):
        frame = pd.read_csv(self.one_way_path)
        frame.loc[len(frame)] = [None, "low"]
        frame.to_csv(self.one_way_path, index=False)
        result = analyze_one_way(self.one_way_config())["outcome"]
        self.assertEqual(result.complete_n, 12)
        self.assertEqual(result.excluded_n, 1)

    def test_group_level_lost_to_missing_responses_is_reported(self):
        self.one_way_path.write_text(
            "outcome,group\n1,A\n2,A\n3,B\n4,B\n,C\n,C\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "no complete response observations"):
            analyze_one_way(
                self.one_way_config(
                    input_path=self.one_way_path,
                    responses=("outcome",),
                    factor_a="group",
                )
            )

    def test_invalid_non_numeric_response_is_rejected(self):
        frame = pd.read_csv(self.one_way_path)
        frame["outcome"] = frame["outcome"].astype(object)
        frame.loc[0, "outcome"] = "not-a-number"
        frame.to_csv(self.one_way_path, index=False)
        with self.assertRaisesRegex(ValueError, "must contain numeric values"):
            analyze_one_way(self.one_way_config())

    def test_duplicate_csv_headers_are_rejected(self):
        self.one_way_path.write_text("outcome,group,group\n1,a,a\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicated column headers"):
            load_csv(self.one_way_path)

    def test_literal_na_category_is_not_treated_as_missing(self):
        self.one_way_path.write_text(
            "outcome,group\n1,NA\n2,NA\n3,other\n4,other\n",
            encoding="utf-8",
        )
        result = analyze_one_way(self.one_way_config())["outcome"]
        self.assertEqual(result.complete_n, 4)
        self.assertEqual(set(result.group_summary["group"]), {"NA", "other"})

    def test_two_way_detects_interaction_and_reports_type_iii(self):
        config = TwoWayConfig(
            input_path=self.two_way_path,
            responses=("outcome",),
            factor_a="factor_a",
            factor_b="factor_b",
            sum_of_squares="III",
            independent_observations_confirmed=True,
        )
        result = analyze_two_way(config)["outcome"]
        self.assertEqual(result.complete_n, 16)
        self.assertLess(result.anova_table.loc["interaction", "PR(>F)"], 0.05)
        self.assertIn("partial_eta_squared", result.anova_table.columns)
        self.assertTrue(result.model_description.startswith("outcome ~"))
        self.assertIn("factor_a within", " ".join(result.posthoc["effect"].tolist()))
        figures = generate_plots([result], self.root / "two_way_output")
        outputs = write_reports(
            [result],
            self.root / "two_way_output",
            self.two_way_path,
            pd.read_csv(self.two_way_path),
            figures,
        )
        self.assertTrue(Path(outputs["pdf_report"]).is_file())
        self.assertTrue((self.root / "two_way_output" / "anova_results.xlsx").is_file())

    def test_empty_factor_combinations_are_rejected(self):
        frame = pd.read_csv(self.two_way_path)
        frame = frame[~((frame["factor_a"] == "A1") & (frame["factor_b"] == "B1"))]
        frame.to_csv(self.two_way_path, index=False)
        config = TwoWayConfig(
            input_path=self.two_way_path,
            responses=("outcome",),
            factor_a="factor_a",
            factor_b="factor_b",
            independent_observations_confirmed=True,
        )
        with self.assertRaisesRegex(ValueError, "empty cell"):
            analyze_two_way(config)

    def test_unbalanced_two_way_design_is_supported(self):
        frame = pd.read_csv(self.two_way_path).drop(index=0)
        frame.to_csv(self.two_way_path, index=False)
        config = TwoWayConfig(
            input_path=self.two_way_path,
            responses=("outcome",),
            factor_a="factor_a",
            factor_b="factor_b",
            sum_of_squares="II",
            independent_observations_confirmed=True,
        )
        result = analyze_two_way(config)["outcome"]
        self.assertEqual(result.complete_n, 15)
        self.assertIn("factor_a", result.anova_table.index)

    def test_all_sums_of_squares_types_are_available(self):
        for sum_of_squares in ("I", "II", "III"):
            config = TwoWayConfig(
                input_path=self.two_way_path,
                responses=("outcome",),
                factor_a="factor_a",
                factor_b="factor_b",
                sum_of_squares=sum_of_squares,
                independent_observations_confirmed=True,
            )
            result = analyze_two_way(config)["outcome"]
            self.assertIn("interaction", result.anova_table.index)
            self.assertIn(sum_of_squares, result.model_description)

    def test_figures_and_research_reports_are_generated(self):
        results = analyze_one_way(self.one_way_config())
        figures = generate_plots(list(results.values()), self.root / "output")
        outputs = write_reports(
            list(results.values()),
            self.root / "output",
            self.one_way_path,
            pd.read_csv(self.one_way_path),
            figures,
        )
        self.assertTrue(Path(outputs["workbook"]).is_file())
        self.assertTrue(Path(outputs["pdf_report"]).is_file())
        self.assertTrue(Path(outputs["manifest"]).is_file())
        self.assertEqual(len(figures["outcome"]), 5)
        self.assertTrue((self.root / "output" / "outcome_posthoc.csv").is_file())


if __name__ == "__main__":
    unittest.main()
