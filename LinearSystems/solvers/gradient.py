"""Residual-direction gradient methods."""

import numpy as np

from ..model import FloatVector, LinearSystem, SolverResult, SolverStatus, StoppingCriteria
from ._common import (
    IterationMonitor,
    criteria_or_default,
    dangerous_denominator,
    initial_vector,
    is_spd,
    unavailable_result,
)


def solve_steepest_descent(
    system: LinearSystem,
    x0: FloatVector | None = None,
    criteria: StoppingCriteria | None = None,
) -> SolverResult:
    """Minimize the SPD quadratic using the optimal residual step."""
    if not is_spd(system):
        return unavailable_result("steepest_descent", "steepest descent requires an SPD matrix")
    return _solve_residual_direction(system, x0, criteria, method="steepest_descent", minimal=False)


def solve_minimal_residual(
    system: LinearSystem,
    x0: FloatVector | None = None,
    criteria: StoppingCriteria | None = None,
) -> SolverResult:
    """Minimize the residual norm along the current residual direction."""
    return _solve_residual_direction(system, x0, criteria, method="minimal_residual", minimal=True)


def _solve_residual_direction(
    system: LinearSystem,
    x0: FloatVector | None,
    criteria: StoppingCriteria | None,
    *,
    method: str,
    minimal: bool,
) -> SolverResult:
    monitor = IterationMonitor(
        method,
        system,
        criteria_or_default(criteria),
        initial_vector(system, x0),
    )
    initial = monitor.initial_result()
    if initial is not None:
        return initial
    assert monitor.solution is not None and monitor.residual is not None
    solution = monitor.solution.copy()
    residual = monitor.residual.copy()

    for iteration in range(1, monitor.criteria.max_iter + 1):
        image = monitor.matvec(residual)
        if not np.all(np.isfinite(image)):
            return monitor.result(SolverStatus.NONFINITE, "A times residual contains NaN or Inf")
        if minimal:
            numerator = float(image @ residual)
            denominator = float(image @ image)
            scale = float(np.linalg.norm(image)) ** 2
        else:
            numerator = float(residual @ residual)
            denominator = float(residual @ image)
            scale = float(np.linalg.norm(residual)) * float(np.linalg.norm(image))
        if not np.isfinite(numerator) or not np.isfinite(denominator) or not np.isfinite(scale):
            return monitor.result(SolverStatus.NONFINITE, "step scalar is nonfinite")
        if dangerous_denominator(denominator, scale):
            return monitor.result(SolverStatus.BREAKDOWN, "step denominator is zero or numerically dangerous")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            candidate = solution + (numerator / denominator) * residual
        status = monitor.commit(iteration, candidate)
        if status is not None:
            reason = "gradient iterate contains NaN or Inf" if status is SolverStatus.NONFINITE else None
            return monitor.result(status, reason)
        assert monitor.solution is not None and monitor.residual is not None
        solution = monitor.solution.copy()
        residual = monitor.residual.copy()
    return monitor.max_iter_result()
