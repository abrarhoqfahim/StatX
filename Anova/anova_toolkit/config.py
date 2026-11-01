"""Shared configuration for ANOVA analyses."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class AnalysisConfig:
    """Inputs and analysis choices shared by the public analysis functions."""

    input_path: Path
    responses: Tuple[str, ...]
    factor_a: str
    factor_b: Optional[str] = None
    alpha: float = 0.05
    sum_of_squares: str = "II"
    independent_observations_confirmed: bool = False

    def __post_init__(self) -> None:
        if not self.responses:
            raise ValueError("Select at least one numeric response column.")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        if self.sum_of_squares not in {"I", "II", "III"}:
            raise ValueError("sum_of_squares must be I, II, or III.")
        if len(set(self.responses)) != len(self.responses):
            raise ValueError("Response column names must be unique.")


@dataclass(frozen=True)
class OneWayConfig(AnalysisConfig):
    """Configuration for one-way ANOVA."""

    method: str = "classical"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.method not in {"classical", "welch", "kruskal"}:
            raise ValueError("method must be classical, welch, or kruskal.")


@dataclass(frozen=True)
class TwoWayConfig(AnalysisConfig):
    """Configuration for two-way ANOVA."""

    factor_b: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.factor_b:
            raise ValueError("factor_b is required for two-way ANOVA.")
        if self.factor_a == self.factor_b:
            raise ValueError("Choose two different factor columns.")
