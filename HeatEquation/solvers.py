import numpy as np

from .discrete_problem import (
    apply_boundary_conditions,
    apply_interior_operator,
    initial_field,
    interior_residual,
    residual_l2,
    residual_max,
)
from .model import (
    FloatArray,
    HeatProblem,
    IterationPoint,
    SolverResult,
    SolverStatus,
    StoppingCriteria,
    StoppingRule,
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


def solve_cg(
    problem: HeatProblem,
    criteria: StoppingCriteria | None = None,
    initial: FloatArray | None = None,
) -> SolverResult:
    stopping = StoppingCriteria() if criteria is None else criteria
    if not isinstance(stopping, StoppingCriteria):
        raise TypeError("criteria must be StoppingCriteria")
    field = _prepare_initial(problem, initial)
    residual_vector = interior_residual(problem, field)
    residual = float(np.max(np.abs(residual_vector)))
    initial_residual_l2 = residual_l2(problem, field)
    initial_relative_residual = 0.0 if initial_residual_l2 == 0.0 else 1.0
    history = [
        IterationPoint(0, residual, initial_relative_residual, None, None)
    ]
    if not np.isfinite(residual):
        return _result(problem, "cg", field, SolverStatus.NONFINITE, 0, history, 1.0)
    if _converged(stopping, residual, initial_relative_residual):
        return _result(problem, "cg", field, SolverStatus.CONVERGED, 0, history, 1.0)

    direction = residual_vector.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        residual_squared = float(np.vdot(residual_vector, residual_vector))
    if not np.isfinite(residual_squared):
        return _result(problem, "cg", field, SolverStatus.NONFINITE, 0, history, 1.0)
    if residual_squared <= 0.0:
        return _result(problem, "cg", field, SolverStatus.BREAKDOWN, 0, history, 1.0)

    completed_iterations = 0
    status = SolverStatus.MAX_ITER
    for iteration in range(1, stopping.max_iterations + 1):
        applied = apply_interior_operator(problem, direction)
        if not np.all(np.isfinite(applied)):
            status = SolverStatus.NONFINITE
            break

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            denominator = float(np.vdot(direction, applied))
        if not np.isfinite(denominator):
            status = SolverStatus.NONFINITE
            break
        if _invalid_cg_denominator(direction, applied, denominator):
            status = SolverStatus.BREAKDOWN
            break

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            step = residual_squared / denominator
        if not np.isfinite(step):
            status = SolverStatus.NONFINITE
            break

        previous = field.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            next_residual_vector = residual_vector - step * applied
            candidate = field[1:-1, 1:-1] + step * direction
        if not np.all(np.isfinite(candidate)):
            status = SolverStatus.NONFINITE
            break
        field[1:-1, 1:-1] = candidate
        completed_iterations = iteration

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            delta = float(np.max(np.abs(field - previous)))
            norm = float(np.max(np.abs(field)))
            relative_delta = delta / max(norm, np.finfo(np.float64).eps)
        residual = residual_max(problem, field)
        relative_residual_l2 = residual_l2(problem, field) / initial_residual_l2
        history.append(
            IterationPoint(
                iteration, residual, relative_residual_l2, delta, relative_delta
            )
        )
        if not np.all(np.isfinite((delta, relative_delta, residual))) or not np.all(
            np.isfinite(next_residual_vector)
        ):
            status = SolverStatus.NONFINITE
            break
        if _converged(stopping, residual, relative_residual_l2):
            status = SolverStatus.CONVERGED
            break

        with np.errstate(over="ignore", invalid="ignore"):
            next_residual_squared = float(
                np.vdot(next_residual_vector, next_residual_vector)
            )
        if not np.isfinite(next_residual_squared):
            status = SolverStatus.NONFINITE
            break
        if next_residual_squared <= 0.0:
            status = SolverStatus.BREAKDOWN
            break
        beta = next_residual_squared / residual_squared
        if not np.isfinite(beta):
            status = SolverStatus.NONFINITE
            break
        with np.errstate(over="ignore", invalid="ignore"):
            direction = next_residual_vector + beta * direction
        if not np.all(np.isfinite(direction)):
            status = SolverStatus.NONFINITE
            break
        residual_vector = next_residual_vector
        residual_squared = next_residual_squared

    return _result(
        problem, "cg", field, status, completed_iterations, history, 1.0
    )


def solve_bicgstab(
    problem: HeatProblem,
    criteria: StoppingCriteria | None = None,
    initial: FloatArray | None = None,
) -> SolverResult:
    """Solve the interior Dirichlet system with matrix-free BiCGSTAB."""
    stopping = StoppingCriteria() if criteria is None else criteria
    if not isinstance(stopping, StoppingCriteria):
        raise TypeError("criteria must be StoppingCriteria")
    field = _prepare_initial(problem, initial)
    residual_vector = interior_residual(problem, field)
    residual = float(np.max(np.abs(residual_vector)))
    initial_residual_l2 = residual_l2(problem, field)
    initial_relative_residual = 0.0 if initial_residual_l2 == 0.0 else 1.0
    history = [
        IterationPoint(0, residual, initial_relative_residual, None, None)
    ]
    if not np.isfinite(residual):
        return _result(
            problem, "bicgstab", field, SolverStatus.NONFINITE, 0, history, 1.0
        )
    if _converged(stopping, residual, initial_relative_residual):
        return _result(
            problem, "bicgstab", field, SolverStatus.CONVERGED, 0, history, 1.0
        )

    shadow = residual_vector.copy()
    direction = residual_vector.copy()
    applied = np.zeros_like(residual_vector)
    rho_previous = 1.0
    alpha = 1.0
    omega = 1.0
    completed_iterations = 0
    status = SolverStatus.MAX_ITER

    for iteration in range(1, stopping.max_iterations + 1):
        rho = _finite_dot(shadow, residual_vector)
        if rho is None:
            status = SolverStatus.NONFINITE
            break
        if _dangerous_dot(shadow, residual_vector, rho):
            status = SolverStatus.BREAKDOWN
            break

        if iteration > 1:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                beta = (rho / rho_previous) * (alpha / omega)
                direction = residual_vector + beta * (direction - omega * applied)
            if not np.isfinite(beta) or not np.all(np.isfinite(direction)):
                status = SolverStatus.NONFINITE
                break

        applied = apply_interior_operator(problem, direction)
        if not np.all(np.isfinite(applied)):
            status = SolverStatus.NONFINITE
            break
        alpha_denominator = _finite_dot(shadow, applied)
        if alpha_denominator is None:
            status = SolverStatus.NONFINITE
            break
        if _dangerous_dot(shadow, applied, alpha_denominator):
            status = SolverStatus.BREAKDOWN
            break

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            alpha = rho / alpha_denominator
            intermediate_residual = residual_vector - alpha * applied
            intermediate_interior = field[1:-1, 1:-1] + alpha * direction
        if not np.isfinite(alpha) or not np.all(
            np.isfinite((intermediate_residual, intermediate_interior))
        ):
            status = SolverStatus.NONFINITE
            break

        previous = field.copy()
        intermediate_field = field.copy()
        intermediate_field[1:-1, 1:-1] = intermediate_interior
        point = _iteration_point(
            problem, intermediate_field, previous, initial_residual_l2, iteration
        )
        if not _finite_point(point):
            status = SolverStatus.NONFINITE
            break
        if _converged(stopping, point.residual, point.relative_residual_l2):
            field = intermediate_field
            history.append(point)
            completed_iterations = iteration
            status = SolverStatus.CONVERGED
            break

        image_s = apply_interior_operator(problem, intermediate_residual)
        if not np.all(np.isfinite(image_s)):
            status = SolverStatus.NONFINITE
            break
        omega_denominator = _finite_dot(image_s, image_s)
        if omega_denominator is None:
            status = SolverStatus.NONFINITE
            break
        if _dangerous_dot(image_s, image_s, omega_denominator):
            status = SolverStatus.BREAKDOWN
            break
        omega_numerator = _finite_dot(image_s, intermediate_residual)
        if omega_numerator is None:
            status = SolverStatus.NONFINITE
            break
        if _dangerous_dot(image_s, intermediate_residual, omega_numerator):
            status = SolverStatus.BREAKDOWN
            break

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            omega = omega_numerator / omega_denominator
            candidate_interior = intermediate_interior + omega * intermediate_residual
            next_residual = intermediate_residual - omega * image_s
        if not np.isfinite(omega) or not np.all(
            np.isfinite((candidate_interior, next_residual))
        ):
            status = SolverStatus.NONFINITE
            break

        field[1:-1, 1:-1] = candidate_interior
        point = _iteration_point(
            problem, field, previous, initial_residual_l2, iteration
        )
        if not _finite_point(point):
            field = previous
            status = SolverStatus.NONFINITE
            break
        history.append(point)
        completed_iterations = iteration
        if _converged(stopping, point.residual, point.relative_residual_l2):
            status = SolverStatus.CONVERGED
            break

        residual_vector = next_residual
        rho_previous = rho

    return _result(
        problem,
        "bicgstab",
        field,
        status,
        completed_iterations,
        history,
        1.0,
    )


def _finite_dot(left: FloatArray, right: FloatArray) -> float | None:
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.vdot(left, right))
    return value if np.isfinite(value) else None


def _dangerous_dot(left: FloatArray, right: FloatArray, value: float) -> bool:
    if value == 0.0:
        return True
    left_scale = float(np.max(np.abs(left)))
    right_scale = float(np.max(np.abs(right)))
    if left_scale == 0.0 or right_scale == 0.0:
        return value == 0.0
    scaled_left = left / left_scale
    scaled_right = right / right_scale
    scaled_value = float(np.vdot(scaled_left, scaled_right))
    scale = float(np.linalg.norm(scaled_left) * np.linalg.norm(scaled_right))
    return (
        not np.isfinite(scaled_value)
        or not np.isfinite(scale)
        or abs(scaled_value) <= np.finfo(np.float64).eps * scale
    )


def _invalid_cg_denominator(
    direction: FloatArray, applied: FloatArray, denominator: float
) -> bool:
    if denominator <= 0.0:
        return True
    direction_scale = float(np.max(np.abs(direction)))
    applied_scale = float(np.max(np.abs(applied)))
    if direction_scale == 0.0 or applied_scale == 0.0:
        return True
    scaled_direction = direction / direction_scale
    scaled_applied = applied / applied_scale
    scaled_denominator = float(np.vdot(scaled_direction, scaled_applied))
    norm_product = float(
        np.linalg.norm(scaled_direction) * np.linalg.norm(scaled_applied)
    )
    return (
        not np.isfinite(scaled_denominator)
        or not np.isfinite(norm_product)
        or scaled_denominator <= np.finfo(np.float64).eps * norm_product
    )


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
    initial_residual_l2 = residual_l2(problem, field)
    initial_relative_residual = 0.0 if initial_residual_l2 == 0.0 else 1.0
    history = [
        IterationPoint(0, residual, initial_relative_residual, None, None)
    ]
    if not np.isfinite(residual):
        return _result(problem, method, field, SolverStatus.NONFINITE, 0, history, omega)
    if _converged(stopping, residual, initial_relative_residual):
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
        relative_residual_l2 = residual_l2(problem, field) / initial_residual_l2
        history.append(
            IterationPoint(
                iteration, residual, relative_residual_l2, delta, relative_delta
            )
        )
        if not np.all(np.isfinite((delta, relative_delta, residual))):
            status = SolverStatus.NONFINITE
            break
        if _converged(stopping, residual, relative_residual_l2):
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


def _converged(
    criteria: StoppingCriteria, residual: float, relative_residual_l2: float
) -> bool:
    value = (
        residual
        if criteria.rule is StoppingRule.ABSOLUTE_MAX
        else relative_residual_l2
    )
    return value < criteria.tolerance


def _iteration_point(
    problem: HeatProblem,
    field: FloatArray,
    previous: FloatArray,
    initial_residual_l2: float,
    iteration: int,
) -> IterationPoint:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = float(np.max(np.abs(field - previous)))
        norm = float(np.max(np.abs(field)))
        relative_delta = delta / max(norm, np.finfo(np.float64).eps)
        relative_residual_l2 = residual_l2(problem, field) / initial_residual_l2
    return IterationPoint(
        iteration,
        residual_max(problem, field),
        relative_residual_l2,
        delta,
        relative_delta,
    )


def _finite_point(point: IterationPoint) -> bool:
    return bool(
        np.all(
            np.isfinite(
                (
                    point.residual,
                    point.relative_residual_l2,
                    point.delta,
                    point.relative_delta,
                )
            )
        )
    )


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
