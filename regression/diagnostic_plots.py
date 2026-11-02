"""Create model diagnostics for fitted ordinary least-squares regressions."""

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def save_diagnostic_plots(
    output_path: Path,
    fitted_model: Any,
    predictor_values: np.ndarray,
    target_values: np.ndarray,
    predictor_names: Sequence[str],
    model_type: str,
) -> None:
    """Save fit, residual, normality, predictor, and influence plots."""
    fitted_values = np.asarray(fitted_model.fittedvalues, dtype=float)
    residuals = np.asarray(fitted_model.resid, dtype=float)
    cooks_distance = np.asarray(
        fitted_model.get_influence().cooks_distance[0], dtype=float
    )
    plot_count = 5 + len(predictor_names)
    columns = 2
    rows = (plot_count + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(12, 4.2 * rows), squeeze=False
    )
    flat_axes = axes.ravel()
    axis_index = 0

    try:
        observed_axis = flat_axes[axis_index]
        axis_index += 1
        observed_axis.scatter(
            fitted_values, target_values, alpha=0.75, label="Observations"
        )
        lower = min(float(np.min(fitted_values)), float(np.min(target_values)))
        upper = max(float(np.max(fitted_values)), float(np.max(target_values)))
        observed_axis.plot(
            [lower, upper],
            [lower, upper],
            "r--",
            linewidth=1,
            label="Perfect prediction",
        )
        observed_axis.set(
            title="Observed vs. predicted",
            xlabel="Predicted values",
            ylabel="Observed values",
        )
        observed_axis.legend()

        residual_axis = flat_axes[axis_index]
        axis_index += 1
        residual_axis.scatter(
            fitted_values, residuals, alpha=0.75, label="Residuals"
        )
        residual_axis.axhline(
            0.0,
            color="red",
            linestyle="--",
            linewidth=1,
            label="Zero residual",
        )
        residual_axis.set(
            title="Residuals vs. fitted values",
            xlabel="Fitted values",
            ylabel="Residuals",
        )
        residual_axis.legend()

        qq_axis = flat_axes[axis_index]
        axis_index += 1
        stats.probplot(residuals, dist="norm", plot=qq_axis)
        qq_axis.set_title("Normal Q-Q plot of residuals")
        qq_lines = qq_axis.get_lines()
        if qq_lines:
            qq_lines[0].set_label("Residual quantiles")
        if len(qq_lines) > 1:
            qq_lines[1].set_label("Normal reference line")
        qq_axis.legend()

        influence_axis = flat_axes[axis_index]
        axis_index += 1
        observation_numbers = np.arange(1, len(cooks_distance) + 1)
        influence_axis.stem(
            observation_numbers,
            cooks_distance,
            markerfmt=",",
            basefmt=" ",
            label="Cook's distance",
        )
        influence_axis.axhline(
            4.0 / len(cooks_distance),
            color="red",
            linestyle="--",
            linewidth=1,
            label="4 / n reference",
        )
        influence_axis.set(
            title="Cook's distance (reference is a heuristic)",
            xlabel="Observation",
            ylabel="Cook's distance",
        )
        influence_axis.legend()

        if len(predictor_names) == 1:
            predictor_axis = flat_axes[axis_index]
            axis_index += 1
            predictor = predictor_values[:, 0]
            order = np.argsort(predictor)
            predictor_axis.scatter(
                predictor, target_values, alpha=0.75, label="Observed"
            )
            predictor_axis.plot(
                predictor[order],
                fitted_values[order],
                color="red",
                linewidth=2,
                label="Fitted",
            )
            predictor_axis.set(
                title="{} model fit".format(model_type.title()),
                xlabel=predictor_names[0],
                ylabel="Response",
            )
            predictor_axis.legend()

        for predictor_index, predictor_name in enumerate(predictor_names):
            predictor_axis = flat_axes[axis_index]
            axis_index += 1
            predictor_axis.scatter(
                predictor_values[:, predictor_index],
                residuals,
                alpha=0.75,
                label="Residuals",
            )
            predictor_axis.axhline(
                0.0,
                color="red",
                linestyle="--",
                linewidth=1,
                label="Zero residual",
            )
            predictor_axis.set(
                title="Residuals vs. {}".format(predictor_name),
                xlabel=predictor_name,
                ylabel="Residuals",
            )
            predictor_axis.legend()

        for unused_axis in flat_axes[axis_index:]:
            figure.delaxes(unused_axis)

        figure.suptitle("{} regression diagnostics".format(model_type.title()))
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
