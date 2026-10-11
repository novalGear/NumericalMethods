import csv
import json
import math
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .model import ExperimentResults


_METRICS_FIELDS = (
    "method",
    "variant",
    "N",
    "omega",
    "iterations",
    "matvec_count",
    "diagnostic_matvec_count",
    "elapsed_median_s",
    "residual_abs",
    "residual_rel",
    "backward_error",
    "reference_error_abs",
    "reference_error_rel",
    "status",
)
_HISTORY_FIELDS = (
    "method",
    "omega",
    "iteration",
    "residual_abs",
    "residual_rel",
    "reference_error_rel",
    "matvec_count",
)
_SOR_SCAN_FIELDS = (
    "omega",
    "iterations",
    "elapsed_median_s",
    "residual_rel",
    "status",
)


def write_experiment(
    results: ExperimentResults,
    output_dir: str | PathLike[str],
) -> tuple[Path, ...]:
    """Write all required tabular and machine-readable experiment data."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    metrics_path = directory / "slau_metrics.csv"
    _write_csv(
        metrics_path,
        _METRICS_FIELDS,
        (
            {
                "method": metric.method,
                "variant": results.system.variant,
                "N": results.system.size,
                "omega": metric.omega,
                "iterations": metric.iterations,
                "matvec_count": metric.matvec_count,
                "diagnostic_matvec_count": metric.diagnostic_matvec_count,
                "elapsed_median_s": metric.elapsed_median_s,
                "residual_abs": metric.residual_abs,
                "residual_rel": metric.residual_rel,
                "backward_error": metric.backward_error,
                "reference_error_abs": metric.reference_error_abs,
                "reference_error_rel": metric.reference_error_rel,
                "status": metric.status.value,
            }
            for metric in results.metrics
        ),
    )

    reference_norm = float(np.linalg.norm(results.reference_solution))
    history_path = directory / "slau_history.csv"
    _write_csv(
        history_path,
        _HISTORY_FIELDS,
        (
            {
                "method": solver_result.method,
                "omega": solver_result.omega,
                "iteration": point.iteration,
                "residual_abs": point.residual_abs,
                "residual_rel": point.residual_rel,
                "reference_error_rel": _reference_error(
                    point.solution, results.reference_solution, reference_norm
                ),
                "matvec_count": point.matvec_count,
            }
            for solver_result in results.solver_results
            for point in solver_result.history
        ),
    )

    sor_scan_path = directory / "slau_sor_scan.csv"
    _write_csv(
        sor_scan_path,
        _SOR_SCAN_FIELDS,
        (
            {
                "omega": point.omega,
                "iterations": point.iterations,
                "elapsed_median_s": point.elapsed_median_s,
                "residual_rel": point.residual_rel,
                "status": point.status.value,
            }
            for point in results.sor_scan
        ),
    )

    solution_path = directory / "slau_solution.csv"
    solution_fields = ("component", "reference") + tuple(
        solver_result.method for solver_result in results.solver_results
    )
    _write_csv(
        solution_path,
        solution_fields,
        (
            {
                "component": component + 1,
                "reference": results.reference_solution[component],
                **{
                    solver_result.method: (
                        ""
                        if solver_result.solution is None
                        else solver_result.solution[component]
                    )
                    for solver_result in results.solver_results
                },
            }
            for component in range(results.system.size)
        ),
    )

    diagnostics_path = directory / "matrix_diagnostics.json"
    diagnostics = results.diagnostics
    with diagnostics_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "size": diagnostics.size,
                "symmetric": diagnostics.symmetric,
                "diagonal_dominance_margin": _json_number(
                    diagnostics.diagonal_dominance_margin
                ),
                "eigenvalue_min": _json_number(diagnostics.eigenvalue_min),
                "eigenvalue_max": _json_number(diagnostics.eigenvalue_max),
                "condition_number_2": _json_number(diagnostics.condition_number_2),
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")

    system_path = directory / "system.npz"
    np.savez(system_path, A=results.system.matrix, b=results.system.rhs)

    return (
        metrics_path,
        history_path,
        sor_scan_path,
        solution_path,
        diagnostics_path,
        system_path,
    )


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: Any,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _reference_error(
    solution: np.ndarray,
    reference: np.ndarray,
    reference_norm: float,
) -> float:
    error = float(np.linalg.norm(solution - reference))
    return error / reference_norm if reference_norm != 0.0 else error


def _json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None
