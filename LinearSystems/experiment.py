from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import numpy as np

from .model import (
    ExperimentResults,
    FloatVector,
    LinearSystem,
    SolverMetrics,
    SolverResult,
    SolverStatus,
    SorScanPoint,
    StoppingCriteria,
    readonly_vector,
)
from .solvers import (
    solve_bicgstab,
    solve_cg,
    solve_gauss_seidel,
    solve_gaussian,
    solve_jacobi,
    solve_lu,
    solve_minimal_residual,
    solve_sor,
    solve_steepest_descent,
)
from .systems import diagnose_system


def default_sor_omegas() -> tuple[float, ...]:
    regular_grid = (1.0,) + tuple(1.0 + step / 100 for step in range(5, 100, 5))
    return tuple(sorted((*regular_grid, 1.01)))


@dataclass(frozen=True)
class ExperimentConfig:
    criteria: StoppingCriteria = StoppingCriteria()
    initial: FloatVector | None = None
    sor_omegas: tuple[float, ...] = default_sor_omegas()
    benchmark_repetitions: int = 30
    benchmark_warmups: int = 1

    def __post_init__(self) -> None:
        if self.initial is not None:
            object.__setattr__(self, "initial", readonly_vector(self.initial))
        if not self.sor_omegas:
            raise ValueError("sor_omegas must not be empty")
        if any(not np.isfinite(omega) or not 0.0 < omega < 2.0 for omega in self.sor_omegas):
            raise ValueError("every SOR omega must be finite and satisfy 0 < omega < 2")
        if len(set(self.sor_omegas)) != len(self.sor_omegas):
            raise ValueError("sor_omegas must not contain duplicates")
        if (
            isinstance(self.benchmark_repetitions, bool)
            or not isinstance(self.benchmark_repetitions, int)
            or self.benchmark_repetitions < 1
        ):
            raise ValueError("benchmark_repetitions must be a positive integer")
        if (
            isinstance(self.benchmark_warmups, bool)
            or not isinstance(self.benchmark_warmups, int)
            or self.benchmark_warmups < 0
        ):
            raise ValueError("benchmark_warmups must be a nonnegative integer")


def run_experiment(
    system: LinearSystem,
    config: ExperimentConfig | None = None,
) -> ExperimentResults:
    configuration = ExperimentConfig() if config is None else config
    initial = (
        np.zeros(system.size, dtype=np.float64)
        if configuration.initial is None
        else np.array(configuration.initial, dtype=np.float64, copy=True)
    )
    if initial.shape != system.rhs.shape:
        raise ValueError(f"initial approximation must have shape {system.rhs.shape}")

    diagnostics = diagnose_system(system)
    reference = np.linalg.solve(system.matrix, system.rhs)

    sor_runs: dict[float, tuple[SolverResult, float]] = {}
    sor_scan: list[SorScanPoint] = []
    for omega in configuration.sor_omegas:
        result, elapsed = _benchmark(
            lambda omega=omega: solve_sor(
                system,
                omega,
                initial,
                configuration.criteria,
            ),
            configuration,
        )
        sor_runs[omega] = (result, elapsed)
        sor_scan.append(
            SorScanPoint(
                omega=omega,
                iterations=(
                    result.iterations
                    if result.status is SolverStatus.CONVERGED
                    else None
                ),
                elapsed_median_s=elapsed,
                residual_rel=_residual_relative(system, result.solution),
                status=result.status,
            )
        )

    converged_scan = [
        point
        for point in sor_scan
        if point.status is SolverStatus.CONVERGED
        and point.iterations is not None
    ]
    if not converged_scan:
        raise RuntimeError("no tested SOR parameter converged")
    best_iteration_count = min(point.iterations for point in converged_scan)
    best_sor_omegas = tuple(
        point.omega
        for point in converged_scan
        if point.iterations == best_iteration_count
    )
    over_relaxed = [point for point in converged_scan if point.omega > 1.0]
    if not over_relaxed:
        raise RuntimeError("no tested SOR parameter above 1 converged")
    selected_iteration_count = min(point.iterations for point in over_relaxed)
    selected_omega = next(
        point.omega
        for point in over_relaxed
        if point.iterations == selected_iteration_count
    )

    configured_solvers: tuple[tuple[str, Callable[[], SolverResult]], ...] = (
        ("gaussian", lambda: solve_gaussian(system)),
        ("lu", lambda: solve_lu(system)),
        (
            "jacobi",
            lambda: solve_jacobi(system, initial, configuration.criteria),
        ),
        (
            "gauss_seidel",
            lambda: solve_gauss_seidel(system, initial, configuration.criteria),
        ),
        (
            "steepest_descent",
            lambda: solve_steepest_descent(system, initial, configuration.criteria),
        ),
        (
            "minimal_residual",
            lambda: solve_minimal_residual(system, initial, configuration.criteria),
        ),
        ("cg", lambda: solve_cg(system, initial, configuration.criteria)),
        (
            "bicgstab",
            lambda: solve_bicgstab(system, initial, configuration.criteria),
        ),
    )

    solver_results: list[SolverResult] = []
    elapsed_by_method: dict[str, float] = {}
    for method, operation in configured_solvers:
        result, elapsed = _benchmark(operation, configuration)
        solver_results.append(result)
        elapsed_by_method[method] = elapsed

    selected_sor, selected_sor_elapsed = sor_runs[selected_omega]
    solver_results.insert(4, selected_sor)
    elapsed_by_method["sor"] = selected_sor_elapsed

    metrics = tuple(
        _calculate_metrics(
            system,
            result,
            reference,
            elapsed_by_method[result.method],
        )
        for result in solver_results
    )
    rhs_norm = float(np.linalg.norm(system.rhs))
    threshold = configuration.criteria.atol + configuration.criteria.rtol * rhs_norm
    threshold_relative = threshold / rhs_norm if rhs_norm != 0.0 else threshold

    return ExperimentResults(
        system=system,
        diagnostics=diagnostics,
        reference_solution=reference,
        solver_results=tuple(solver_results),
        metrics=metrics,
        sor_scan=tuple(sor_scan),
        best_sor_omegas=best_sor_omegas,
        threshold_relative=threshold_relative,
    )


def _benchmark(
    operation: Callable[[], SolverResult],
    config: ExperimentConfig,
) -> tuple[SolverResult, float]:
    for _ in range(config.benchmark_warmups):
        operation()

    durations: list[float] = []
    result: SolverResult | None = None
    for _ in range(config.benchmark_repetitions):
        started = perf_counter()
        current = operation()
        durations.append(perf_counter() - started)
        if result is None:
            result = current

    assert result is not None
    return result, float(median(durations))


def _calculate_metrics(
    system: LinearSystem,
    result: SolverResult,
    reference: FloatVector,
    elapsed: float,
) -> SolverMetrics:
    if result.solution is None:
        residual_abs = residual_rel = backward_error = float("nan")
        reference_error_abs = reference_error_rel = float("nan")
    else:
        residual = system.rhs - system.matrix @ result.solution
        residual_abs = float(np.linalg.norm(residual))
        rhs_norm = float(np.linalg.norm(system.rhs))
        residual_rel = residual_abs / rhs_norm if rhs_norm != 0.0 else residual_abs
        denominator = (
            float(np.linalg.norm(system.matrix, ord=2))
            * float(np.linalg.norm(result.solution))
            + rhs_norm
        )
        backward_error = residual_abs / denominator if denominator != 0.0 else residual_abs
        reference_error_abs = float(np.linalg.norm(result.solution - reference))
        reference_norm = float(np.linalg.norm(reference))
        reference_error_rel = (
            reference_error_abs / reference_norm
            if reference_norm != 0.0
            else reference_error_abs
        )

    return SolverMetrics(
        method=result.method,
        omega=result.omega,
        iterations=result.iterations,
        matvec_count=result.matvec_count,
        diagnostic_matvec_count=result.diagnostic_matvec_count,
        elapsed_median_s=elapsed,
        residual_abs=residual_abs,
        residual_rel=residual_rel,
        backward_error=backward_error,
        reference_error_abs=reference_error_abs,
        reference_error_rel=reference_error_rel,
        status=result.status,
    )


def _residual_relative(
    system: LinearSystem,
    solution: FloatVector | None,
) -> float:
    if solution is None:
        return float("nan")
    residual_abs = float(np.linalg.norm(system.rhs - system.matrix @ solution))
    rhs_norm = float(np.linalg.norm(system.rhs))
    return residual_abs / rhs_norm if rhs_norm != 0.0 else residual_abs
