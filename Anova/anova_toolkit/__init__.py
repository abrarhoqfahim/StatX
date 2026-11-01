"""Research-oriented one-way and two-way ANOVA toolkit."""

from .config import AnalysisConfig
from .one_way_anova import analyze_one_way
from .two_way_anova import analyze_two_way

__all__ = ["AnalysisConfig", "analyze_one_way", "analyze_two_way"]
