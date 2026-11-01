import logging
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .model import ExperimentResults, SolverStatus


LOGGER = logging.getLogger(__name__)

_HISTORY_STYLES = (
    ("o", "-"),
    ("s", "--"),
    ("^", "-."),
    ("D", ":"),
    ("v", "-"),
    ("P", "--"),
    ("X", "-."),
)


def save_experiment_plots(
    results: ExperimentResults,
    output_dir: str | PathLike[str],
) -> tuple[Path, ...]:
    """Save the two required linear-system figures as PNG and PDF files."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    history_results = tuple(
        solver_result
        for solver_result in results.solver_results
        if solver_result.history
    )
    if len(history_results) != 7:
        raise ValueError(
            "expected histories for exactly seven iterative methods, "
            f"got {len(history_results)}"
        )

    output_paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(9, 6))
    try:
        omitted_zero_count = 0
        for solver_result, (marker, linestyle) in zip(
            history_results, _HISTORY_STYLES, strict=True
        ):
            iterations = np.fromiter(
                (point.iteration for point in solver_result.history), dtype=np.int64
            )
            residuals = np.fromiter(
                (point.residual_rel for point in solver_result.history),
                dtype=np.float64,
            )
            plottable = np.isfinite(residuals) & (residuals > 0.0)
            zero_count = int(np.count_nonzero(residuals == 0.0))
            invalid_count = int(np.count_nonzero(~np.isfinite(residuals) | (residuals < 0.0)))
            if zero_count:
                omitted_zero_count += zero_count
                LOGGER.warning(
                    "Omitted %d exact zero residual(s) from log plot for %s",
                    zero_count,
                    solver_result.method,
                )
            if invalid_count:
                LOGGER.warning(
                    "Omitted %d invalid residual(s) from log plot for %s",
                    invalid_count,
                    solver_result.method,
                )

            label = solver_result.method
            if solver_result.omega is not None:
                label = f"{label} (omega={solver_result.omega:g})"
            axis.semilogy(
                iterations[plottable],
                residuals[plottable],
                marker=marker,
                linestyle=linestyle,
                markersize=4,
                markevery=max(1, int(np.count_nonzero(plottable)) // 15),
                label=label,
            )

        if omitted_zero_count:
            axis.text(
                0.01,
                0.01,
                f"Zero-residual points omitted: {omitted_zero_count}",
                transform=axis.transAxes,
                fontsize=8,
                verticalalignment="bottom",
            )

        if np.isfinite(results.threshold_relative) and results.threshold_relative > 0.0:
            axis.axhline(
                results.threshold_relative,
                color="black",
                linestyle=":",
                linewidth=1.4,
                label="Stopping threshold",
            )
        else:
            LOGGER.warning(
                "Did not plot nonpositive or nonfinite relative threshold %r on log axis",
                results.threshold_relative,
            )

        axis.set_xlabel("Iteration, k")
        axis.set_ylabel("Relative true residual")
        axis.set_title(
            f"Iterative solver residual histories (variant {results.system.variant})"
        )
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
        figure.tight_layout()
        output_paths.extend(
            _save_figure(figure, directory / "slau_residual_vs_iteration")
        )
    finally:
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    try:
        converged = tuple(
            point
            for point in results.sor_scan
            if point.status is SolverStatus.CONVERGED and point.iterations is not None
        )
        if converged:
            axis.plot(
                [point.omega for point in converged],
                [point.iterations for point in converged],
                marker="o",
                linestyle="-",
                label="Converged",
            )

        failed_by_status: dict[str, list[float]] = {}
        for point in results.sor_scan:
            if point.status is SolverStatus.CONVERGED and point.iterations is not None:
                continue
            status = point.status.value
            if point.status is SolverStatus.CONVERGED:
                status = "converged (iterations unavailable)"
            failed_by_status.setdefault(status, []).append(point.omega)
        for status, omegas in failed_by_status.items():
            axis.scatter(
                omegas,
                np.full(len(omegas), 0.98),
                transform=axis.get_xaxis_transform(),
                marker="x",
                s=55,
                linewidths=1.5,
                clip_on=False,
                label=status,
            )

        axis.axvline(1.0, color="gray", linestyle=":", label="omega=1")
        for index, omega in enumerate(results.best_sor_omegas):
            axis.axvline(
                omega,
                color="tab:red",
                linestyle="--",
                alpha=0.75,
                label="Best tested omega" if index == 0 else None,
            )
        axis.set_xlabel("Relaxation parameter, omega")
        axis.set_ylabel("Iterations to convergence")
        axis.set_title(f"SOR scan (variant {results.system.variant})")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        output_paths.extend(
            _save_figure(figure, directory / "slau_sor_iterations_vs_omega")
        )
    finally:
        plt.close(figure)

    return tuple(output_paths)


def _save_figure(figure: plt.Figure, path_without_suffix: Path) -> tuple[Path, Path]:
    png_path = path_without_suffix.with_suffix(".png")
    pdf_path = path_without_suffix.with_suffix(".pdf")
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path)
    return png_path, pdf_path
