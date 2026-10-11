"""Direct solvers for dense linear systems."""

import numpy as np

from ..model import LinearSystem, SolverResult, SolverStatus
from ._common import dangerous_denominator, is_spd


def _direct_failure(method: str, status: SolverStatus, reason: str) -> SolverResult:
    return SolverResult(method=method, solution=None, status=status, iterations=None, reason=reason)


def _direct_success(
    method: str,
    system: LinearSystem,
    solution: np.ndarray,
    diagnostics: tuple[tuple[str, float], ...] = (),
) -> SolverResult:
    with np.errstate(over="ignore", invalid="ignore"):
        residual = system.rhs - system.matrix @ solution
    if not np.all(np.isfinite(solution)) or not np.all(np.isfinite(residual)):
        return SolverResult(
            method=method,
            solution=None,
            status=SolverStatus.NONFINITE,
            iterations=None,
            diagnostic_matvec_count=1,
            reason="solution or true residual contains NaN or Inf",
            diagnostics=diagnostics,
        )
    return SolverResult(
        method=method,
        solution=solution,
        status=SolverStatus.CONVERGED,
        iterations=None,
        diagnostic_matvec_count=1,
        diagnostics=diagnostics,
    )


def solve_gaussian(system: LinearSystem) -> SolverResult:
    """Solve by Gaussian elimination with partial pivoting."""
    method = "gaussian"
    matrix = np.array(system.matrix, dtype=np.float64, copy=True)
    rhs = np.array(system.rhs, dtype=np.float64, copy=True)
    size = system.size
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for column in range(size - 1):
            pivot_row = column + int(np.argmax(np.abs(matrix[column:, column])))
            pivot = float(matrix[pivot_row, column])
            column_scale = float(np.max(np.abs(matrix[column:, column])))
            if dangerous_denominator(pivot, column_scale):
                return _direct_failure(method, SolverStatus.BREAKDOWN, f"dangerous pivot at column {column}")
            if pivot_row != column:
                matrix[[column, pivot_row]] = matrix[[pivot_row, column]]
                rhs[[column, pivot_row]] = rhs[[pivot_row, column]]
            factors = matrix[column + 1 :, column] / matrix[column, column]
            matrix[column + 1 :, column:] -= factors[:, None] * matrix[column, column:]
            rhs[column + 1 :] -= factors * rhs[column]
            if not np.all(np.isfinite(matrix[column + 1 :, column:])) or not np.all(
                np.isfinite(rhs[column + 1 :])
            ):
                return _direct_failure(method, SolverStatus.NONFINITE, "nonfinite elimination state")

        solution = np.empty(size, dtype=np.float64)
        for row in range(size - 1, -1, -1):
            pivot = float(matrix[row, row])
            row_scale = float(np.max(np.abs(matrix[row, row:])))
            if dangerous_denominator(pivot, row_scale):
                return _direct_failure(method, SolverStatus.BREAKDOWN, f"dangerous pivot at row {row}")
            solution[row] = (rhs[row] - matrix[row, row + 1 :] @ solution[row + 1 :]) / pivot

    return _direct_success(method, system, solution)


def solve_lu(system: LinearSystem) -> SolverResult:
    """Solve an SPD system using Doolittle LU factorization without pivoting."""
    method = "lu"
    if not is_spd(system):
        return _direct_failure(method, SolverStatus.NOT_APPLICABLE, "Doolittle solver requires an SPD matrix")

    matrix = system.matrix
    size = system.size
    lower = np.eye(size, dtype=np.float64)
    upper = np.zeros_like(matrix)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for row in range(size):
            upper[row, row:] = matrix[row, row:] - lower[row, :row] @ upper[:row, row:]
            pivot = float(upper[row, row])
            column_scale = float(np.max(np.abs(matrix[row:, row])))
            if dangerous_denominator(pivot, column_scale):
                return _direct_failure(method, SolverStatus.BREAKDOWN, f"dangerous LU pivot at row {row}")
            if row + 1 < size:
                lower[row + 1 :, row] = (
                    matrix[row + 1 :, row] - lower[row + 1 :, :row] @ upper[:row, row]
                ) / pivot
            if not np.all(np.isfinite(upper[row, row:])) or not np.all(
                np.isfinite(lower[row + 1 :, row])
            ):
                return _direct_failure(method, SolverStatus.NONFINITE, "nonfinite LU factorization state")

        intermediate = np.empty(size, dtype=np.float64)
        for row in range(size):
            intermediate[row] = system.rhs[row] - lower[row, :row] @ intermediate[:row]

        solution = np.empty(size, dtype=np.float64)
        for row in range(size - 1, -1, -1):
            solution[row] = (
                intermediate[row] - upper[row, row + 1 :] @ solution[row + 1 :]
            ) / upper[row, row]

    factorization_error = float(np.linalg.norm(matrix - lower @ upper, ord=2)) / float(
        np.linalg.norm(matrix, ord=2)
    )
    return _direct_success(
        method,
        system,
        solution,
        diagnostics=(("relative_factorization_error_2", factorization_error),),
    )
