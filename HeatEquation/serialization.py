import csv
import json
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .convergence import GridConvergenceStudy
from .experiments import MethodComparison, SolverScalingStudy, SorScan


def write_experiment(
    comparison: MethodComparison,
    sor_scan: SorScan,
    output_dir: str | PathLike[str],
    grid_convergence: GridConvergenceStudy | None = None,
    smooth_verification: GridConvergenceStudy | None = None,
    solver_scaling: SolverScalingStudy | None = None,
) -> tuple[Path, ...]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    metrics_path = directory / "heat_methods.csv"
    _write_csv(
        metrics_path,
        (
            "method",
            "omega",
            "iterations",
            "final_residual",
            "elapsed_seconds",
            "status",
        ),
        (
            {
                "method": run.result.method,
                "omega": run.result.omega,
                "iterations": run.result.iterations,
                "final_residual": run.result.final_residual,
                "elapsed_seconds": run.elapsed_seconds,
                "status": run.result.status.value,
            }
            for run in comparison.runs
        ),
    )

    history_path = directory / "heat_history.csv"
    _write_csv(
        history_path,
        (
            "method",
            "iteration",
            "residual",
            "relative_residual_l2",
            "delta",
            "relative_delta",
        ),
        (
            {
                "method": run.result.method,
                "iteration": point.iteration,
                "residual": point.residual,
                "relative_residual_l2": point.relative_residual_l2,
                "delta": point.delta,
                "relative_delta": point.relative_delta,
            }
            for run in comparison.runs
            for point in run.result.history
        ),
    )

    scan_path = directory / "heat_sor_scan.csv"
    _write_csv(
        scan_path,
        ("omega", "iterations", "final_residual", "elapsed_seconds", "status"),
        (
            {
                "omega": point.omega,
                "iterations": point.iterations,
                "final_residual": point.final_residual,
                "elapsed_seconds": point.elapsed_seconds,
                "status": point.status.value,
            }
            for point in sor_scan.points
        ),
    )

    fields_path = directory / "heat_fields.npz"
    np.savez(
        fields_path,
        x=comparison.problem.x,
        y=comparison.problem.y,
        **{run.result.method: run.result.field for run in comparison.runs},
    )

    summary_path = directory / "heat_summary.json"
    summary: dict[str, Any] = {
        "problem": {
            "lx": comparison.problem.lx,
            "ly": comparison.problem.ly,
            "nx": comparison.problem.nx,
            "ny": comparison.problem.ny,
            "hx": comparison.problem.hx,
            "hy": comparison.problem.hy,
        },
        "stopping_criteria": {
            "tolerance": comparison.criteria.tolerance,
            "max_iterations": comparison.criteria.max_iterations,
            "rule": comparison.criteria.rule.value,
        },
        "methods": [
            {
                "method": run.result.method,
                "omega": run.result.omega,
                "iterations": run.result.iterations,
                "final_residual": run.result.final_residual,
                "elapsed_seconds": run.elapsed_seconds,
                "status": run.result.status.value,
            }
            for run in comparison.runs
        ],
        "pairwise_differences": [
            difference.__dict__ for difference in comparison.pairwise_differences
        ],
        "best_sor_omega": sor_scan.best_omega,
    }
    paths = [metrics_path, history_path, scan_path, fields_path, summary_path]

    if grid_convergence is not None:
        path = directory / "heat_grid_convergence.csv"
        _write_convergence(path, grid_convergence)
        summary["ordinary_grid_convergence"] = _convergence_summary(grid_convergence)
        paths.append(path)
    if smooth_verification is not None:
        path = directory / "heat_smooth_verification.csv"
        _write_convergence(path, smooth_verification)
        summary["smooth_exact_verification"] = _convergence_summary(
            smooth_verification
        )
        paths.append(path)
    if solver_scaling is not None:
        path = directory / "heat_solver_scaling.csv"
        records = _scaling_records(solver_scaling)
        _write_csv(
            path,
            (
                "ny",
                "nx",
                "unknowns",
                "method",
                "status",
                "iterations",
                "elapsed_seconds",
                "final_absolute_max_residual",
                "final_relative_residual_l2",
                "stopping_rule",
                "tolerance",
                "max_iterations",
            ),
            records,
        )
        summary["solver_scaling"] = {
            "sor_omega": solver_scaling.sor_omega,
            "stopping_criteria": {
                "tolerance": solver_scaling.criteria.tolerance,
                "max_iterations": solver_scaling.criteria.max_iterations,
                "rule": solver_scaling.criteria.rule.value,
            },
            "points": records,
        }
        paths.append(path)

    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return tuple(paths)


def _write_convergence(path: Path, study: GridConvergenceStudy) -> None:
    _write_csv(
        path,
        (
            "ny",
            "nx",
            "method",
            "h",
            "iterations",
            "final_residual",
            "status",
            "grid_difference",
            "observed_order",
            "exact_error",
            "exact_order",
        ),
        _convergence_records(study),
    )


def _convergence_records(study: GridConvergenceStudy) -> list[dict[str, Any]]:
    return [
        {
            "ny": point.grid[0],
            "nx": point.grid[1],
            "method": study.method,
            "h": point.h,
            "iterations": point.iterations,
            "final_residual": point.final_residual,
            "status": point.status.value,
            "grid_difference": point.grid_difference,
            "observed_order": point.observed_order,
            "exact_error": point.exact_error,
            "exact_order": point.exact_order,
        }
        for point in study.points
    ]


def _convergence_summary(study: GridConvergenceStudy) -> dict[str, Any]:
    return {"method": study.method, "points": _convergence_records(study)}


def _scaling_records(study: SolverScalingStudy) -> list[dict[str, Any]]:
    return [
        {
            "ny": point.grid[0],
            "nx": point.grid[1],
            "unknowns": point.unknowns,
            "method": point.method,
            "status": point.status.value,
            "iterations": point.iterations,
            "elapsed_seconds": point.elapsed_seconds,
            "final_absolute_max_residual": point.final_absolute_max_residual,
            "final_relative_residual_l2": point.final_relative_residual_l2,
            "stopping_rule": study.criteria.rule.value,
            "tolerance": study.criteria.tolerance,
            "max_iterations": study.criteria.max_iterations,
        }
        for point in study.points
    ]


def _write_csv(path: Path, fields: tuple[str, ...], rows: Any) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
