"""Stationary iterative solvers."""

import numpy as np

from ..model import FloatVector, LinearSystem, SolverResult, SolverStatus, StoppingCriteria
from ._common import IterationMonitor, criteria_or_default, initial_vector, unavailable_result


def _valid_diagonal(system: LinearSystem) -> bool:
    diagonal = np.abs(np.diag(system.matrix))
    row_scales = np.max(np.abs(system.matrix), axis=1)
    return bool(np.all(diagonal > np.finfo(np.float64).eps * row_scales))


def _setup(
    method: str,
    system: LinearSystem,
    x0: FloatVector | None,
    criteria: StoppingCriteria | None,
    *,
    omega: float | None = None,
) -> tuple[IterationMonitor | None, SolverResult | None]:
    if not _valid_diagonal(system):
        return None, unavailable_result(method, "matrix diagonal contains a zero or dangerous value", omega=omega)
    monitor = IterationMonitor(
        method,
        system,
        criteria_or_default(criteria),
        initial_vector(system, x0),
        omega=omega,
    )
    return monitor, monitor.initial_result()


def solve_jacobi(
    system: LinearSystem,
    x0: FloatVector | None = None,
    criteria: StoppingCriteria | None = None,
) -> SolverResult:
    """Solve using simultaneous Jacobi component updates."""
    monitor, result = _setup("jacobi", system, x0, criteria)
    if result is not None:
        return result
    assert monitor is not None and monitor.solution is not None
    diagonal = np.diag(system.matrix)
    remainder = system.matrix - np.diag(diagonal)
    solution = monitor.solution.copy()

    for iteration in range(1, monitor.criteria.max_iter + 1):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            candidate = (system.rhs - monitor.matvec(solution, remainder)) / diagonal
        status = monitor.commit(iteration, candidate)
        if status is not None:
            reason = "Jacobi update produced NaN or Inf" if status is SolverStatus.NONFINITE else None
            return monitor.result(status, reason)
        solution = candidate
    return monitor.max_iter_result()


def solve_gauss_seidel(
    system: LinearSystem,
    x0: FloatVector | None = None,
    criteria: StoppingCriteria | None = None,
) -> SolverResult:
    """Solve using sequential Gauss-Seidel component updates."""
    return _solve_sequential(system, x0, criteria, method="gauss_seidel", omega=1.0, expose_omega=False)


def solve_sor(
    system: LinearSystem,
    omega: float,
    x0: FloatVector | None = None,
    criteria: StoppingCriteria | None = None,
) -> SolverResult:
    """Solve using successive over-relaxation with ``0 < omega < 2``."""
    if not np.isfinite(omega) or not 0.0 < omega < 2.0:
        return unavailable_result("sor", "SOR requires a finite omega with 0 < omega < 2", omega=float(omega))
    return _solve_sequential(system, x0, criteria, method="sor", omega=float(omega), expose_omega=True)


def _solve_sequential(
    system: LinearSystem,
    x0: FloatVector | None,
    criteria: StoppingCriteria | None,
    *,
    method: str,
    omega: float,
    expose_omega: bool,
) -> SolverResult:
    monitor, result = _setup(method, system, x0, criteria, omega=omega if expose_omega else None)
    if result is not None:
        return result
    assert monitor is not None and monitor.solution is not None
    matrix = system.matrix
    diagonal = np.diag(matrix)
    solution = monitor.solution.copy()

    for iteration in range(1, monitor.criteria.max_iter + 1):
        old = solution.copy()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for row in range(system.size):
                gs_value = (
                    system.rhs[row]
                    - matrix[row, :row] @ solution[:row]
                    - matrix[row, row + 1 :] @ old[row + 1 :]
                ) / diagonal[row]
                solution[row] = (1.0 - omega) * old[row] + omega * gs_value
        status = monitor.commit(iteration, solution)
        if status is not None:
            reason = "sequential update produced NaN or Inf" if status is SolverStatus.NONFINITE else None
            return monitor.result(status, reason)
    return monitor.max_iter_result()
