"""Common analysis result container."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class AnalysisResult:
    """Tabular outputs and reproducibility metadata for one response."""

    response: str
    analysis: str
    complete_n: int
    excluded_n: int
    alpha: float
    model_description: str
    group_summary: pd.DataFrame
    anova_table: pd.DataFrame
    assumptions: pd.DataFrame
    posthoc: pd.DataFrame
    interpretation: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
