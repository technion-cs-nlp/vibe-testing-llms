"""
Evaluation package initialization.

This package now hosts both the original `Evaluator` helper (kept in
`legacy.py` for backwards compatibility) and the new multi-paradigm
evaluation strategies introduced for code and patch validation flows.
"""

from src.vibe_testing.evaluation.legacy import Evaluator

__all__ = ["Evaluator"]


