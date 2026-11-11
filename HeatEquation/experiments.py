from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .discrete_problem import initial_field
from .model import (
    BoundaryConditions,
    HeatProblem,
    SolverResult,
    SolverStatus,
    StoppingCriteria,
    StoppingRule,
)
from .solvers import (
    solve_bicgstab,
    solve_cg,
    solve_gauss_seidel,
    solve_jacobi,
    solve_sor,
)


@dataclass(frozen=True)
class TimedSolverRun:
    result: SolverResult
    elapsed_seconds: float


@dataclass(frozen=True)
class PairwiseDifference:
    first: str
    second: str
    max_difference: float


@dataclass(frozen=True)
class MethodComparison:
    problem: HeatProblem
    criteria: StoppingCriteria
    runs: tuple[TimedSolverRun, ...]
    pairwise_differences: tuple[PairwiseDifference, ...]


@dataclass(frozen=True)
class SorScanPoint:
    omega: float
    iterations: int
    final_residual: float
    elapsed_seconds: float
    status: SolverStatus

    @property
    def converged(self) -> bool:
        return self.status is SolverStatus.CONVERGED


@dataclass(frozen=True)
class SorScan:
    points: tuple[SorScanPoint, ...]
    best_omega: float | None


@dataclass(frozen=True)
class SolverScalingPoint:
    grid: tuple[int, int]
    unknowns: int
    method: str
    status: SolverStatus
    iterations: int
    elapsed_seconds: float
    final_absolute_max_residual: float
    final_relative_residual_l2: float


@dataclass(frozen=True)
class SolverScalingStudy:
    criteria: StoppingCriteria
    sor_omega: float
    points: tuple[SolverScalingPoint, ...]


def compare_methods(
    problem: HeatProblem,
    criteria: StoppingCriteria | None = None,
    omega: float = 1.7,
) -> MethodComparison:
    stopping = StoppingCriteria() if criteria is None else criteria
    operations = (
        lambda: solve_jacobi(problem, stopping),
        lambda: solve_gauss_seidel(problem, stopping),
        lambda: solve_sor(problem, omega, stopping),
        lambda: solve_cg(problem, stopping),
        lambda: solve_bicgstab(problem, stopping),
    )
    runs = tuple(_timed(operation) for operation in operations)
    incomplete = [run.result.method for run in runs if not run.result.converged]
    if incomplete:
        methods = ", ".join(incomplete)
        raise RuntimeError(f"cannot compare incomplete solver runs: {methods}")
    differences = tuple(
        PairwiseDifference(
            first.result.method,
            second.result.method,
            float(np.max(np.abs(first.result.field - second.result.field))),
        )
        for index, first in enumerate(runs)
        for second in runs[index + 1 :]
    )
    return MethodComparison(problem, stopping, runs, differences)


def scan_sor(
    problem: HeatProblem,
    omegas: tuple[float, ...] | list[float],
    criteria: StoppingCriteria | None = None,
) -> SorScan:
    stopping = StoppingCriteria() if criteria is None else criteria
    values = tuple(float(omega) for omega in omegas)
    if not values:
        raise ValueError("omegas must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("omegas must not contain duplicates")

    points: list[SorScanPoint] = []
    for omega in values:
        run = _timed(lambda omega=omega: solve_sor(problem, omega, stopping))
        result = run.result
        points.append(
            SorScanPoint(
                omega=omega,
                iterations=result.iterations,
                final_residual=result.final_residual,
                elapsed_seconds=run.elapsed_seconds,
                status=result.status,
            )
        )
        # Only scalar data in SorScanPoint survives this iteration.

    converged = [point for point in points if point.converged]
    best = min(converged, key=lambda point: point.iterations).omega if converged else None
    return SorScan(tuple(points), best)


def run_solver_scaling(
    nx: int = 5,
    ny: int = 5,
    levels: int = 4,
    criteria: StoppingCriteria | None = None,
    sor_omega: float = 1.7,
) -> SolverScalingStudy:
    """Measure all solvers on nested smooth harmonic grids using SOR omega 1.7."""
    if isinstance(levels, bool) or not isinstance(levels, int):
        raise TypeError("levels must be an integer")
    if levels < 1:
        raise ValueError("levels must be positive")
    stopping = (
        StoppingCriteria(1e-8, 100_000, StoppingRule.RELATIVE_L2)
        if criteria is None
        else criteria
    )
    exact = lambda x, y: float(np.exp(x) * np.cos(y))
    boundaries = BoundaryConditions(
        left=lambda y: exact(0.0, y),
        right=lambda y: exact(1.0, y),
        bottom=lambda x: exact(x, 0.0),
        top=lambda x: exact(x, 1.0),
    )
    points: list[SolverScalingPoint] = []
    for _ in range(levels):
        problem = HeatProblem(1.0, 1.0, nx, ny, boundaries)
        initial = initial_field(problem)
        operations = (
            lambda: solve_jacobi(problem, stopping, initial),
            lambda: solve_gauss_seidel(problem, stopping, initial),
            lambda: solve_sor(problem, sor_omega, stopping, initial),
            lambda: solve_cg(problem, stopping, initial),
            lambda: solve_bicgstab(problem, stopping, initial),
        )
        for operation in operations:
            run = _timed(operation)
            result = run.result
            points.append(
                SolverScalingPoint(
                    grid=(problem.ny, problem.nx),
                    unknowns=(problem.nx - 2) * (problem.ny - 2),
                    method=result.method,
                    status=result.status,
                    iterations=result.iterations,
                    elapsed_seconds=run.elapsed_seconds,
                    final_absolute_max_residual=result.final_residual,
                    final_relative_residual_l2=(
                        result.history[-1].relative_residual_l2
                    ),
                )
            )
        nx, ny = 2 * (nx - 1) + 1, 2 * (ny - 1) + 1
    return SolverScalingStudy(stopping, sor_omega, tuple(points))


def _timed(operation: Callable[[], SolverResult]) -> TimedSolverRun:
    started = perf_counter()
    result = operation()
    return TimedSolverRun(result, perf_counter() - started)
