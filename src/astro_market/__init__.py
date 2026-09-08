"""Astro Market Alpha: walk-forward evaluation of astrological indicator rules."""

__version__ = "0.1.0"

from astro_market.evaluate import evaluate_rule, fitness
from astro_market.rules import parse_rule, evaluate_ast

__all__ = [
    "__version__",
    "evaluate_rule",
    "fitness",
    "parse_rule",
    "evaluate_ast",
]
