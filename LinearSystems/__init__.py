"""Numerical methods for dense linear systems."""

from .model import LinearSystem, SolverResult, SolverStatus, StoppingCriteria
from .systems import build_variant_l, diagnose_system

__all__ = [
    "LinearSystem",
    "SolverResult",
    "SolverStatus",
    "StoppingCriteria",
    "build_variant_l",
    "diagnose_system",
]
