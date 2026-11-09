import numpy as np

from .discrete_problem import apply_boundary_conditions, initial_field, residual_max
from .model import (
    FloatArray,
    HeatProblem,
    IterationPoint,
    SolverResult,
    SolverStatus,
    StoppingCriteria,
)


def solve_jacobi(
    problem: HeatProblem,
    criteria: StoppingCriteria | None = None,
    initial: FloatArray | None = None,
) -> SolverResult:
    return _solve(problem, criteria, initial, "jacobi", jacobi=True, omega=1.0)


def solve_gauss_seidel(
    problem: HeatProblem,
    criteria: StoppingCriteria | None = None,
    initial: FloatArray | None = None,
) -> SolverResult:
    return _solve(
        problem, criteria, initial, "gauss_seidel", jacobi=False, omega=1.0
    )


def solve_sor(
    problem: HeatProblem,
    omega: float,
    criteria: StoppingCriteria | None = None,
    initial: FloatArray | None = None,
) -> SolverResult:
    if not np.isfinite(omega) or not 0.0 < omega < 2.0:
        raise ValueError("omega must be finite and satisfy 0 < omega < 2")
    return _solve(problem, criteria, initial, "sor", jacobi=False, omega=float(omega))


def _solve(
    problem: HeatProblem,
    criteria: StoppingCriteria | None,
    initial: FloatArray | None,
    method: str,
    *,
    jacobi: bool,
    omega: float,
) -> SolverResult:
    stopping = StoppingCriteria() if criteria is None else criteria
    if not isinstance(stopping, StoppingCriteria):
        raise TypeError("criteria must be StoppingCriteria")
    field = _prepare_initial(problem, initial)

    # These normalized weights are algebraically equivalent to inverse-square
    # coefficients but avoid overflow in the update formula.
    if problem.hx >= problem.hy:
        ratio = (problem.hy / problem.hx) ** 2
        weight_x = ratio / (2.0 * (1.0 + ratio))
        weight_y = 1.0 / (2.0 * (1.0 + ratio))
    else:
        ratio = (problem.hx / problem.hy) ** 2
        weight_x = 1.0 / (2.0 * (1.0 + ratio))
        weight_y = ratio / (2.0 * (1.0 + ratio))

    residual = residual_max(problem, field)
    history = [IterationPoint(0, residual, None, None)]
    if not np.isfinite(residual):
        return _result(problem, method, field, SolverStatus.NONFINITE, 0, history, omega)
    if residual < stopping.tolerance:
        return _result(problem, method, field, SolverStatus.CONVERGED, 0, history, omega)

    completed_iterations = 0
    status = SolverStatus.MAX_ITER
    for iteration in range(1, stopping.max_iterations + 1):
        previous = field.copy()
        finite_update = True
        with np.errstate(over="ignore", invalid="ignore"):
            if jacobi:
                candidate = (
                    weight_x * previous[1:-1, :-2]
                    + weight_x * previous[1:-1, 2:]
                    + weight_y * previous[:-2, 1:-1]
                    + weight_y * previous[2:, 1:-1]
                )
                finite_update = bool(np.all(np.isfinite(candidate)))
                if finite_update:
                    field[1:-1, 1:-1] = candidate
            else:
                for iy in range(1, problem.ny - 1):
                    for ix in range(1, problem.nx - 1):
                        gs_value = (
                            weight_x * field[iy, ix - 1]
                            + weight_x * field[iy, ix + 1]
                            + weight_y * field[iy - 1, ix]
                            + weight_y * field[iy + 1, ix]
                        )
                        candidate = field[iy, ix] + omega * (
                            gs_value - field[iy, ix]
                        )
                        if not np.isfinite(candidate):
                            finite_update = False
                            break
                        field[iy, ix] = candidate
                    if not finite_update:
                        break

        if not finite_update:
            field = previous
            status = SolverStatus.NONFINITE
            break

        completed_iterations = iteration
        with np.errstate(over="ignore", invalid="ignore"):
            delta = float(np.max(np.abs(field - previous)))
            norm = float(np.max(np.abs(field)))
            relative_delta = delta / max(norm, np.finfo(np.float64).eps)
        residual = residual_max(problem, field)
        history.append(IterationPoint(iteration, residual, delta, relative_delta))
        if not np.all(np.isfinite((delta, relative_delta, residual))):
            status = SolverStatus.NONFINITE
            break
        if residual < stopping.tolerance:
            status = SolverStatus.CONVERGED
            break

    return _result(
        problem, method, field, status, completed_iterations, history, omega
    )


def _prepare_initial(problem: HeatProblem, initial: FloatArray | None) -> FloatArray:
    boundary = apply_boundary_conditions(problem)
    if initial is None:
        return initial_field(problem)
    field = np.array(initial, dtype=np.float64, copy=True)
    if field.shape != (problem.ny, problem.nx):
        raise ValueError(f"initial field must have shape {(problem.ny, problem.nx)}")
    if not np.all(np.isfinite(field)):
        raise ValueError("initial field must contain only finite values")
    field[0, :], field[-1, :] = boundary[0, :], boundary[-1, :]
    field[:, 0], field[:, -1] = boundary[:, 0], boundary[:, -1]
    return field


def _result(
    problem: HeatProblem,
    method: str,
    field: FloatArray,
    status: SolverStatus,
    iterations: int,
    history: list[IterationPoint],
    omega: float,
) -> SolverResult:
    final_residual = residual_max(problem, field)
    return SolverResult(
        method=method,
        field=field,
        status=status,
        iterations=iterations,
        final_residual=final_residual,
        history=tuple(history),
        omega=omega if method == "sor" else None,
    )
