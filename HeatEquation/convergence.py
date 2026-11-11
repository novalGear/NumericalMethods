from collections.abc import Callable
from dataclasses import dataclass
from math import log2

import numpy as np

from .model import (
    BoundaryConditions,
    FloatArray,
    HeatProblem,
    SolverResult,
    SolverStatus,
    StoppingCriteria,
    readonly_array,
)
from .solvers import solve_cg, solve_sor


@dataclass(frozen=True)
class GridConvergencePoint:
    grid: tuple[int, int]
    h: float
    iterations: int
    final_residual: float
    status: SolverStatus
    grid_difference: float | None
    observed_order: float | None
    exact_error: float | None
    exact_order: float | None


@dataclass(frozen=True)
class GridFieldSnapshot:
    problem: HeatProblem
    field: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.problem, HeatProblem):
            raise TypeError("problem must be a HeatProblem")
        field = readonly_array(self.field, ndim=2)
        if field.shape != (self.problem.ny, self.problem.nx):
            raise ValueError(
                "field shape must match the snapshot problem grid "
                f"{(self.problem.ny, self.problem.nx)}"
            )
        object.__setattr__(self, "field", field)


@dataclass(frozen=True)
class GridConvergenceStudy:
    study_type: str
    method: str
    points: tuple[GridConvergencePoint, ...]
    snapshots: tuple[GridFieldSnapshot, ...] = ()

    def __post_init__(self) -> None:
        points = tuple(self.points)
        snapshots = tuple(self.snapshots)
        if snapshots and len(snapshots) != len(points):
            raise ValueError("snapshots must contain one field for every grid point")
        if snapshots and any(
            point.grid != (snapshot.problem.ny, snapshot.problem.nx)
            for point, snapshot in zip(points, snapshots, strict=True)
        ):
            raise ValueError("each snapshot grid must match its convergence point")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "snapshots", snapshots)


def run_smooth_harmonic_verification(
    nx: int = 5,
    ny: int = 5,
    levels: int = 5,
    criteria: StoppingCriteria | None = None,
) -> GridConvergenceStudy:
    """Verify second-order accuracy with non-polynomial exp(x) cos(y)."""
    exact = lambda x, y: float(np.exp(x) * np.cos(y))
    problem = HeatProblem(
        1.0,
        1.0,
        nx,
        ny,
        BoundaryConditions(
            left=lambda y: exact(0.0, y),
            right=lambda y: exact(1.0, y),
            bottom=lambda x: exact(x, 0.0),
            top=lambda x: exact(x, 1.0),
        ),
    )
    stopping = criteria or StoppingCriteria(3e-10, 100_000)
    return _run_grid_convergence(
        problem,
        stopping,
        levels,
        exact,
        solve_cg,
        "smooth_exact_verification",
    )


def restrict_to_coarse(fine: FloatArray, coarse_shape: tuple[int, int]) -> FloatArray:
    expected = (2 * (coarse_shape[0] - 1) + 1, 2 * (coarse_shape[1] - 1) + 1)
    if fine.ndim != 2 or fine.shape != expected:
        raise ValueError(
            f"fine grid shape must be {expected} for coarse shape {coarse_shape}"
        )
    restricted = np.array(fine[::2, ::2], copy=True)
    restricted.setflags(write=False)
    return restricted


def run_grid_convergence(
    base_problem: HeatProblem,
    criteria: StoppingCriteria | None = None,
    levels: int = 4,
    omega: float = 1.7,
    exact_solution: Callable[[float, float], float] | None = None,
    *,
    retain_fields: bool = False,
) -> GridConvergenceStudy:
    stopping = StoppingCriteria(1e-10, 200_000) if criteria is None else criteria
    return _run_grid_convergence(
        base_problem,
        stopping,
        levels,
        exact_solution,
        lambda problem, stopping: solve_sor(problem, omega, stopping),
        "ordinary_grid_convergence",
        retain_fields,
    )


def _run_grid_convergence(
    base_problem: HeatProblem,
    stopping: StoppingCriteria,
    levels: int,
    exact_solution: Callable[[float, float], float] | None,
    solve: Callable[[HeatProblem, StoppingCriteria], SolverResult],
    study_type: str,
    retain_fields: bool = False,
) -> GridConvergenceStudy:
    if isinstance(levels, bool) or not isinstance(levels, int):
        raise TypeError("levels must be an integer")
    if levels < 2:
        raise ValueError("levels must be at least 2")
    problems: list[HeatProblem] = []
    results = []
    nx, ny = base_problem.nx, base_problem.ny
    for _ in range(levels):
        problem = HeatProblem(
            base_problem.lx,
            base_problem.ly,
            nx,
            ny,
            base_problem.boundaries,
        )
        result = solve(problem, stopping)
        if result.status is not SolverStatus.CONVERGED:
            raise RuntimeError(
                f"{result.method} ended with {result.status.value} on grid {(ny, nx)}"
            )
        problems.append(problem)
        results.append(result)
        nx, ny = 2 * (nx - 1) + 1, 2 * (ny - 1) + 1

    differences: list[float | None] = []
    for index, result in enumerate(results):
        if index == len(results) - 1:
            differences.append(None)
        else:
            restricted = restrict_to_coarse(
                results[index + 1].field, result.field.shape
            )
            difference = float(np.max(np.abs(result.field - restricted)))
            differences.append(difference if np.isfinite(difference) else None)

    exact_errors: list[float | None] = []
    for problem, result in zip(problems, results, strict=True):
        if exact_solution is None:
            exact_errors.append(None)
            continue
        try:
            exact = np.array(
                [
                    [exact_solution(float(x), float(y)) for x in problem.x]
                    for y in problem.y
                ],
                dtype=np.float64,
            )
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("exact_solution could not be evaluated") from error
        if not np.all(np.isfinite(exact)):
            raise ValueError("exact_solution produced a non-finite value")
        with np.errstate(over="ignore", invalid="ignore"):
            exact_error = float(np.max(np.abs(result.field - exact)))
        if not np.isfinite(exact_error):
            raise ValueError("exact error is not representable as a finite float")
        exact_errors.append(exact_error)

    points = []
    for index, (problem, result) in enumerate(zip(problems, results, strict=True)):
        observed = (
            _safe_order(differences[index], differences[index + 1])
            if index + 1 < len(differences)
            else None
        )
        exact_order = (
            _safe_order(exact_errors[index], exact_errors[index + 1])
            if index + 1 < len(exact_errors)
            else None
        )
        points.append(
            GridConvergencePoint(
                grid=(problem.ny, problem.nx),
                h=max(problem.hx, problem.hy),
                iterations=result.iterations,
                final_residual=result.final_residual,
                status=result.status,
                grid_difference=differences[index],
                observed_order=observed,
                exact_error=exact_errors[index],
                exact_order=exact_order,
            )
        )
    snapshots = (
        tuple(
            GridFieldSnapshot(problem, result.field)
            for problem, result in zip(problems, results, strict=True)
        )
        if retain_fields
        else ()
    )
    return GridConvergenceStudy(
        study_type,
        results[0].method,
        tuple(points),
        snapshots,
    )


def _safe_order(coarse_error: float | None, fine_error: float | None) -> float | None:
    if coarse_error is None or fine_error is None:
        return None
    if not np.isfinite(coarse_error) or not np.isfinite(fine_error):
        return None
    if coarse_error <= 0.0 or fine_error <= 0.0:
        return None
    ratio = coarse_error / fine_error
    if not np.isfinite(ratio) or ratio <= 0.0:
        return None
    order = log2(ratio)
    return order if np.isfinite(order) else None
