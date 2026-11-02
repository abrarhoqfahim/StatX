"""Tests for the regression CSV command."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import statsmodels.api as sm

from regression.regression_analysis import (
    create_design_matrix,
    read_regression_data,
)


class RegressionAnalysisTests(unittest.TestCase):
    def _write_csv(self, rows):
        temporary_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv", delete=False
        )
        self.addCleanup(Path(temporary_file.name).unlink)
        writer = csv.writer(temporary_file)
        writer.writerows(rows)
        temporary_file.close()
        return Path(temporary_file.name)

    def test_linear_regression_recovers_known_coefficients(self):
        path = self._write_csv(
            [["x", "y"], [0, 2], [1, 5], [2, 8], [3, 11], [4, 14]]
        )
        predictors, target, total, excluded = read_regression_data(
            path, "y", ["x"]
        )
        design, _, _ = create_design_matrix(predictors, ["x"], "linear", None)
        fitted = sm.OLS(target, design).fit()

        np.testing.assert_allclose(fitted.params, [2.0, 3.0], atol=1e-12)
        self.assertEqual(total, 5)
        self.assertEqual(excluded, 0)
        self.assertAlmostEqual(fitted.rsquared, 1.0)

    def test_multiple_regression_fits_multiple_predictors(self):
        path = self._write_csv(
            [
                ["x1", "x2", "y"],
                [0, 0, 1],
                [1, 0, 3],
                [0, 1, 4],
                [1, 1, 6],
                [2, 1, 8],
                [1, 2, 9],
            ]
        )
        predictors, target, _, _ = read_regression_data(
            path, "y", ["x1", "x2"]
        )
        design, _, _ = create_design_matrix(
            predictors, ["x1", "x2"], "multiple", None
        )
        fitted = sm.OLS(target, design).fit()

        np.testing.assert_allclose(fitted.params, [1.0, 2.0, 3.0], atol=1e-12)

    def test_polynomial_regression_uses_standardized_polynomial_terms(self):
        path = self._write_csv(
            [["x", "y"]]
            + [[x, 1 + 2 * x + 3 * x**2] for x in range(-3, 4)]
        )
        predictors, target, _, _ = read_regression_data(path, "y", ["x"])
        design, names, basis = create_design_matrix(
            predictors, ["x"], "polynomial", 2
        )
        fitted = sm.OLS(target, design).fit()

        self.assertEqual(names, ["Intercept", "z^1", "z^2"])
        self.assertIn("z = (x -", basis)
        self.assertAlmostEqual(fitted.rsquared, 1.0)
        self.assertAlmostEqual(float(fitted.mse_resid), 0.0, places=20)

    def test_blank_selected_values_are_excluded(self):
        path = self._write_csv(
            [["x", "y"], [1, 3], [2, ""], ["", 7], [4, 9]]
        )

        predictors, target, total, excluded = read_regression_data(
            path, "y", ["x"]
        )

        self.assertEqual(total, 4)
        self.assertEqual(excluded, 2)
        np.testing.assert_array_equal(predictors[:, 0], [1.0, 4.0])
        np.testing.assert_array_equal(target, [3.0, 9.0])

    def test_non_numeric_selected_value_is_rejected(self):
        path = self._write_csv([["x", "y"], [1, 3], ["bad", 5]])

        with self.assertRaisesRegex(ValueError, "Non-numeric value"):
            read_regression_data(path, "y", ["x"])

    def test_non_numeric_value_is_rejected_even_on_incomplete_row(self):
        path = self._write_csv([["x", "y"], ["bad", ""], [1, 3]])

        with self.assertRaisesRegex(ValueError, "Non-numeric value"):
            read_regression_data(path, "y", ["x"])

    def test_cli_prints_fit_and_coefficient_statistics(self):
        path = self._write_csv(
            [["x", "y"], [0, 2], [1, 5], [2, 8], [3, 11], [4, 14]]
        )
        script = Path(__file__).resolve().parents[1] / "regression_analysis.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                str(path),
                "--target",
                "y",
                "--predictors",
                "x",
                "--model",
                "linear",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("R-squared:", result.stdout)
        self.assertIn("Model F-test p-value:", result.stdout)
        self.assertIn(
            "Term\tEstimate\tStd. error\tt\tp-value\tCI lower\tCI upper",
            result.stdout,
        )
        self.assertIn("x\t3", result.stdout)


if __name__ == "__main__":
    unittest.main()
