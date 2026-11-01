"""Reproducible, publication-quality diagnostic and group plots."""

import re
from pathlib import Path
from typing import Dict, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .results import AnalysisResult


sns.set_theme(style="whitegrid", context="notebook")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "response"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_plots(
    results: Iterable[AnalysisResult], output_dir: Path
) -> Dict[str, list]:
    """Write group, distribution, and model diagnostic figures at 300 DPI."""
    plot_dir = Path(output_dir) / "figures"
    result_list = list(results)
    stems = [_safe_name(result.response) for result in result_list]
    if len(set(stems)) != len(stems):
        raise ValueError("Response names map to duplicate figure filenames.")
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, list] = {}
    for result in result_list:
        frame = result.diagnostics.get("plot_data")
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(
                "Plot data are unavailable for response {!r}.".format(result.response)
            )
        stem = _safe_name(result.response)
        paths = []
        factor_a = result.diagnostics["factor_a"]
        response = result.response

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if result.diagnostics["method"] == "two-way":
            factor_b = result.diagnostics["factor_b"]
            sns.boxplot(data=frame, x=factor_a, y=response, hue=factor_b, ax=axes[0])
            sns.violinplot(
                data=frame,
                x=factor_a,
                y=response,
                hue=factor_b,
                inner="quart",
                cut=0,
                ax=axes[1],
            )
            for axis in axes:
                axis.legend(title=factor_b, bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            sns.boxplot(data=frame, x=factor_a, y=response, ax=axes[0])
            sns.violinplot(data=frame, x=factor_a, y=response, inner="quart", cut=0, ax=axes[1])
        axes[0].set_title("Group distributions")
        axes[1].set_title("Group distributions (violin)")
        fig.suptitle("{}: observed distributions".format(response))
        distribution_path = plot_dir / "{}_distributions.png".format(stem)
        _save(fig, distribution_path)
        paths.append(str(distribution_path))

        fig, axis = plt.subplots(figsize=(8, 5))
        sns.histplot(data=frame, x=response, hue=factor_a, element="step", stat="density", common_norm=False, ax=axis)
        axis.set_title("{}: observed-value distributions".format(response))
        histogram_path = plot_dir / "{}_value_histogram.png".format(stem)
        _save(fig, histogram_path)
        paths.append(str(histogram_path))

        fig, axis = plt.subplots(figsize=(9, 5))
        summary = result.group_summary
        if result.diagnostics["method"] == "two-way":
            factor_b = result.diagnostics["factor_b"]
            factor_a_levels = list(frame[factor_a].cat.categories)
            for level, subset in summary.groupby(factor_b, observed=True, sort=True):
                subset = subset.sort_values(factor_a)
                x = np.arange(len(subset))
                means = subset["mean"].to_numpy(dtype=float)
                lower = means - subset["ci_low"].to_numpy(dtype=float)
                upper = subset["ci_high"].to_numpy(dtype=float) - means
                axis.errorbar(x, means, yerr=np.vstack([lower, upper]), marker="o", capsize=3, label=str(level))
            axis.set_xticks(
                np.arange(summary[factor_a].nunique()),
                [str(value) for value in factor_a_levels],
            )
            axis.set_xlabel(factor_a)
            axis.legend(title=factor_b)
        else:
            summary = summary.sort_values(factor_a)
            means = summary["mean"].to_numpy(dtype=float)
            lower = means - summary["ci_low"].to_numpy(dtype=float)
            upper = summary["ci_high"].to_numpy(dtype=float) - means
            axis.errorbar(
                np.arange(len(summary)),
                means,
                yerr=np.vstack([lower, upper]),
                marker="o",
                capsize=4,
            )
            axis.set_xticks(np.arange(len(summary)), [str(value) for value in summary[factor_a]])
            axis.set_xlabel(factor_a)
        axis.set_ylabel(response)
        confidence_level = (1 - result.alpha) * 100
        axis.set_title(
            "Group means with pointwise t-based {:.1f}% confidence intervals".format(
                confidence_level
            )
        )
        mean_path = plot_dir / "{}_means.png".format(stem)
        _save(fig, mean_path)
        paths.append(str(mean_path))

        fitted = result.diagnostics.get("fitted")
        residuals = result.diagnostics.get("residuals")
        if fitted is not None and residuals is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].scatter(fitted, residuals, alpha=0.65)
            axes[0].axhline(0, color="black", linewidth=1)
            axes[0].set_xlabel("Fitted values")
            axes[0].set_ylabel("Residuals")
            axes[0].set_title("Residuals vs fitted")
            stats.probplot(residuals, dist="norm", plot=axes[1])
            axes[1].set_title("Normal Q-Q plot")
            fig.suptitle("{}: model diagnostics".format(response))
            diagnostic_path = plot_dir / "{}_diagnostics.png".format(stem)
            _save(fig, diagnostic_path)
            paths.append(str(diagnostic_path))

            fig, axis = plt.subplots(figsize=(8, 5))
            sns.histplot(residuals, kde=True, ax=axis)
            axis.set_title("{}: residual distribution".format(response))
            histogram_path = plot_dir / "{}_residual_histogram.png".format(stem)
            _save(fig, histogram_path)
            paths.append(str(histogram_path))

        if result.diagnostics["method"] == "two-way":
            factor_b = result.diagnostics["factor_b"]
            factor_a_levels = list(frame[factor_a].cat.categories)
            fig, axis = plt.subplots(figsize=(9, 5))
            for level, subset in summary.groupby(factor_b, observed=True, sort=True):
                subset = subset.sort_values(factor_a)
                axis.plot(
                    [str(value) for value in factor_a_levels],
                    subset["mean"],
                    marker="o",
                    label=str(level),
                )
            axis.set_xlabel(factor_a)
            axis.set_ylabel("Mean {}".format(response))
            axis.set_title("{} × {} interaction plot".format(factor_a, factor_b))
            axis.legend(title=factor_b)
            interaction_path = plot_dir / "{}_interaction.png".format(stem)
            _save(fig, interaction_path)
            paths.append(str(interaction_path))
        generated[result.response] = paths
    return generated
