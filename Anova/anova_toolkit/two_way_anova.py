"""Two-way factorial ANOVA with explicit sums-of-squares and contrasts."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from .config import TwoWayConfig
from .data_loader import load_csv
from .data_validator import validate_dataset
from .preprocessing import group_descriptives
from .preprocessing import prepare_response
from .results import AnalysisResult
from .statistics import add_effect_sizes, pairwise_tukey, residual_assumptions


def _formula_name(name: str) -> str:
    return "Q({!r})".format(name)


def _simple_effect_groups(
    frame: pd.DataFrame,
    response: str,
    tested_factor: str,
    stratifying_factor: str,
) -> List[Tuple[str, List[Tuple[str, np.ndarray]]]]:
    sets = []
    for stratum, stratum_frame in frame.groupby(
        stratifying_factor, observed=True, sort=True
    ):
        groups = [
            (str(level), values.to_numpy(dtype=float))
            for level, values in stratum_frame.groupby(
                tested_factor, observed=True, sort=True
            )[response]
        ]
        if len(groups) >= 2:
            sets.append(
                (
                    "{} = {}".format(stratifying_factor, stratum),
                    groups,
                )
            )
    return sets


def _posthoc(
    frame: pd.DataFrame,
    response: str,
    factor_a: str,
    factor_b: str,
    table: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    interaction_significant = (
        "interaction" in table.index
        and pd.notna(table.loc["interaction", "PR(>F)"])
        and table.loc["interaction", "PR(>F)"] < alpha
    )
    comparisons = []
    if interaction_significant:
        sets = [
            (factor_a, factor_b, items)
            for items in _simple_effect_groups(frame, response, factor_a, factor_b)
        ] + [
            (factor_b, factor_a, items)
            for items in _simple_effect_groups(frame, response, factor_b, factor_a)
        ]
        for tested_factor, _, (stratum, groups) in sets:
            posthoc = pairwise_tukey(groups, alpha)
            if not posthoc.empty:
                posthoc.insert(0, "effect", "{} within {}".format(tested_factor, stratum))
                comparisons.append(posthoc)
    else:
        for factor in (factor_a, factor_b):
            term = "factor_a" if factor == factor_a else "factor_b"
            if term not in table.index or table.loc[term, "PR(>F)"] >= alpha:
                continue
            groups = [
                (str(level), values.to_numpy(dtype=float))
                for level, values in frame.groupby(factor, observed=True, sort=True)[response]
            ]
            posthoc = pairwise_tukey(groups, alpha)
            if not posthoc.empty:
                posthoc.insert(0, "effect", factor)
                comparisons.append(posthoc)
    return pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame(
        columns=[
            "effect",
            "group_1",
            "group_2",
            "mean_difference",
            "ci_low",
            "ci_high",
            "p_adjusted",
            "reject",
            "note",
        ]
    )


def _analyze_response(
    frame: pd.DataFrame, response: str, config: TwoWayConfig
) -> AnalysisResult:
    factor_a, factor_b = config.factor_a, config.factor_b
    data, excluded = prepare_response(frame, response, (factor_a, factor_b))
    cell_counts = data.groupby(
        [factor_a, factor_b], observed=False, dropna=False
    ).size()
    empty_cells = cell_counts[cell_counts == 0]
    if not empty_cells.empty:
        raise ValueError(
            "Two-way ANOVA requires observations in every factor combination; "
            "empty cell(s): {}.".format(
                ", ".join("{} × {}".format(a, b) for a, b in empty_cells.index)
            )
        )

    formula = "{} ~ C({}, Sum) * C({}, Sum)".format(
        _formula_name(response), _formula_name(factor_a), _formula_name(factor_b)
    )
    model = ols(formula, data=data).fit()
    try:
        table = anova_lm(model, typ={"I": 1, "II": 2, "III": 3}[config.sum_of_squares])
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "The selected factorial model is not estimable for this design: {}".format(error)
        ) from error
    terms = {}
    for term in table.index:
        term_string = str(term)
        if term_string == "Residual":
            terms[term] = "Residual"
        elif ":" in term_string:
            terms[term] = "interaction"
        elif _formula_name(factor_a) in term_string:
            terms[term] = "factor_a"
        elif _formula_name(factor_b) in term_string:
            terms[term] = "factor_b"
        else:
            terms[term] = term_string
    table.index = [terms[index] for index in table.index]
    total_ss = float(np.sum((data[response] - data[response].mean()) ** 2))
    table = add_effect_sizes(table, "Residual", total_ss=total_ss)
    assumptions = residual_assumptions(
        model.resid,
        [
            cell[response].to_numpy(dtype=float)
            for _, cell in data.groupby([factor_a, factor_b], observed=True)
        ],
        config.alpha,
    )
    posthoc = _posthoc(
        data, response, factor_a, factor_b, table, config.alpha
    )

    interpretations = []
    for term, label in (
        ("factor_a", factor_a),
        ("factor_b", factor_b),
        ("interaction", "{} × {}".format(factor_a, factor_b)),
    ):
        if term not in table.index:
            continue
        p_value = float(table.loc[term, "PR(>F)"])
        eta_squared = float(table.loc[term, "partial_eta_squared"])
        if p_value < config.alpha:
            interpretations.append(
                "{}: evidence of an effect (p = {:.4g}, partial eta-squared = {:.4g})."
                .format(label, p_value, eta_squared)
            )
        else:
            interpretations.append(
                "{}: insufficient evidence of an effect (p = {:.4g}).".format(
                    label, p_value
                )
            )
    if table.loc["interaction", "PR(>F)"] < config.alpha:
        interpretations.append(
            "The interaction is significant; interpret simple effects and the "
            "interaction plot rather than isolated marginal effects."
        )
    interpretations.append(
        "Significance does not establish causation or practical importance."
    )

    return AnalysisResult(
        response=response,
        analysis="two-way factorial ANOVA",
        complete_n=len(data),
        excluded_n=excluded,
        alpha=config.alpha,
        model_description=(
            "{} ~ C({}, Sum) * C({}, Sum); Type {} sums of squares; "
            "independent groups.".format(
                response, factor_a, factor_b, config.sum_of_squares
            )
        ),
        group_summary=group_descriptives(
            data, response, (factor_a, factor_b), config.alpha
        ),
        anova_table=table,
        assumptions=assumptions,
        posthoc=posthoc,
        interpretation=interpretations,
        diagnostics={
            "fitted": model.fittedvalues.to_numpy(),
            "residuals": model.resid.to_numpy(),
            "factor_a": factor_a,
            "factor_b": factor_b,
            "method": "two-way",
            "sum_of_squares": config.sum_of_squares,
            "plot_data": data,
        },
    )


def analyze_two_way(config: TwoWayConfig) -> Dict[str, AnalysisResult]:
    """Analyze each response with a two-factor model including interaction."""
    if not config.independent_observations_confirmed:
        raise ValueError(
            "Confirm that rows are independent experimental units before analysis. "
            "Repeated measures require a different model."
        )
    frame = load_csv(config.input_path)
    validate_dataset(frame, config.responses, (config.factor_a, config.factor_b))
    return {
        response: _analyze_response(frame, response, config)
        for response in config.responses
    }
