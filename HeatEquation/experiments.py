from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .model import HeatProblem, SolverResult, SolverStatus, StoppingCriteria
from .solvers import solve_gauss_seidel, solve_jacobi, solve_sor


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


def _timed(operation: Callable[[], SolverResult]) -> TimedSolverRun:
    started = perf_counter()
    result = operation()
    return TimedSolverRun(result, perf_counter() - started)
