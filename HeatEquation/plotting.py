from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .convergence import GridConvergenceStudy
from .experiments import MethodComparison, SorScan
from .model import SolverResult, SolverStatus


def save_experiment_plots(
    comparison: MethodComparison,
    sor_scan: SorScan,
    output_dir: str | PathLike[str],
    grid_convergence: GridConvergenceStudy | None = None,
    smooth_verification: GridConvergenceStudy | None = None,
) -> tuple[Path, ...]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for run in comparison.runs:
        result = run.result
        figure, axis = plt.subplots(figsize=(7, 4.5))
        _plot_positive_history(axis, result, "residual")
        axis.set(
            title=f"{result.method} residual",
            xlabel="Iteration",
            ylabel="Maximum residual",
        )
        axis.grid(True, which="both", alpha=0.3)
        paths.append(
            _save(figure, directory / f"heat_residual_{result.method}.png")
        )

    paths.append(
        _save_history_comparison(
            comparison,
            directory / "heat_residual_all.png",
            "residual",
            "Solver residual convergence",
            "Maximum residual",
        )
    )
    paths.append(
        _save_history_comparison(
            comparison,
            directory / "heat_delta_all.png",
            "delta",
            "Absolute update histories",
            "Maximum absolute update",
        )
    )
    paths.append(
        _save_history_comparison(
            comparison,
            directory / "heat_relative_delta_all.png",
            "relative_delta",
            "Relative update histories",
            "Relative maximum update",
        )
    )

    result = comparison.runs[-1].result
    extent = (0, comparison.problem.lx, 0, comparison.problem.ly)
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(
        result.field, origin="lower", extent=extent, aspect="auto", cmap="inferno"
    )
    figure.colorbar(image, ax=axis, label="Temperature")
    axis.set(title="Temperature field", xlabel="x", ylabel="y")
    paths.append(_save(figure, directory / "heat_field.png"))

    figure, axis = plt.subplots(figsize=(7, 5))
    contours = axis.contour(
        comparison.problem.x, comparison.problem.y, result.field, levels=12
    )
    axis.clabel(contours, inline=True, fontsize=8)
    axis.set(title="Temperature isotherms", xlabel="x", ylabel="y")
    paths.append(_save(figure, directory / "heat_contours.png"))

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for index in sorted({0, comparison.problem.ny // 2, comparison.problem.ny - 1}):
        axes[0].plot(
            comparison.problem.x,
            result.field[index],
            label=f"y={comparison.problem.y[index]:.3g}",
        )
    for index in sorted({0, comparison.problem.nx // 2, comparison.problem.nx - 1}):
        axes[1].plot(
            comparison.problem.y,
            result.field[:, index],
            label=f"x={comparison.problem.x[index]:.3g}",
        )
    axes[0].set(xlabel="x", ylabel="Temperature", title="Horizontal sections")
    axes[1].set(xlabel="y", ylabel="Temperature", title="Vertical sections")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    paths.append(_save(figure, directory / "heat_sections.png"))

    figure, axis = plt.subplots(figsize=(7, 4.5))
    converged = [point for point in sor_scan.points if point.converged]
    failed = [point for point in sor_scan.points if not point.converged]
    if converged:
        axis.plot(
            [point.omega for point in converged],
            [point.iterations for point in converged],
            marker="o",
            label="converged",
        )
    for status in (SolverStatus.MAX_ITER, SolverStatus.NONFINITE):
        points = [point for point in failed if point.status is status]
        if points:
            axis.scatter(
                [point.omega for point in points],
                [point.iterations for point in points],
                marker="x",
                label=status.value,
            )
    if sor_scan.best_omega is not None:
        axis.axvline(
            sor_scan.best_omega,
            color="tab:red",
            linestyle="--",
            label=f"best={sor_scan.best_omega:g}",
        )
    axis.set(title="SOR relaxation scan", xlabel="omega", ylabel="Iterations")
    axis.grid(True, alpha=0.3)
    axis.legend()
    paths.append(_save(figure, directory / "heat_sor_scan.png"))

    if grid_convergence is not None:
        paths.extend(_save_grid_plots(grid_convergence, directory))
    if smooth_verification is not None:
        paths.extend(_save_smooth_plots(smooth_verification, directory))
    return tuple(paths)


def _plot_positive_history(
    axis: plt.Axes, result: SolverResult, attribute: str
) -> None:
    points = result.history
    pairs = [
        (point.iteration, getattr(point, attribute))
        for point in points
        if getattr(point, attribute) is not None
        and np.isfinite(getattr(point, attribute))
        and getattr(point, attribute) > 0.0
    ]
    if pairs:
        axis.semilogy(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            label=result.method,
        )


def _save_history_comparison(
    comparison: MethodComparison,
    path: Path,
    attribute: str,
    title: str,
    ylabel: str,
) -> Path:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for run in comparison.runs:
        _plot_positive_history(axis, run.result, attribute)
    axis.set(title=title, xlabel="Iteration", ylabel=ylabel)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    return _save(figure, path)


def _save_grid_plots(
    study: GridConvergenceStudy, directory: Path
) -> tuple[Path, ...]:
    valid = [
        point
        for point in study.points
        if point.grid_difference is not None and point.grid_difference > 0
    ]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    if valid:
        h = np.array([point.h for point in valid])
        errors = np.array([point.grid_difference for point in valid])
        axis.loglog(h, errors, marker="o", label="E(h)")
        axis.loglog(
            h, errors[0] * (h / h[0]) ** 2, linestyle="--", label="O(h^2)"
        )
    axis.set(title="Ordinary nested-grid convergence", xlabel="h", ylabel="E(h)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    error_path = _save(figure, directory / "heat_grid_convergence.png")

    ordered = [point for point in study.points if point.observed_order is not None]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        [point.h for point in ordered],
        [point.observed_order for point in ordered],
        marker="o",
    )
    axis.axhline(2.0, color="gray", linestyle="--")
    axis.set(title="Ordinary observed grid order", xlabel="h", ylabel="Order p")
    axis.grid(True, alpha=0.3)
    order_path = _save(figure, directory / "heat_grid_order.png")
    return error_path, order_path


def _save_smooth_plots(
    study: GridConvergenceStudy, directory: Path
) -> tuple[Path, ...]:
    valid = [
        point
        for point in study.points
        if point.exact_error is not None and point.exact_error > 0
    ]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    if valid:
        h = np.array([point.h for point in valid])
        errors = np.array([point.exact_error for point in valid])
        axis.loglog(h, errors, marker="o", label="exact error")
        axis.loglog(
            h, errors[0] * (h / h[0]) ** 2, linestyle="--", label="O(h^2)"
        )
    axis.set(title="Smooth harmonic exact error", xlabel="h", ylabel="Exact error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    error_path = _save(figure, directory / "heat_smooth_exact_error.png")

    ordered = [point for point in study.points if point.exact_order is not None]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        [point.h for point in ordered],
        [point.exact_order for point in ordered],
        marker="o",
    )
    axis.axhline(2.0, color="gray", linestyle="--")
    axis.set(title="Smooth harmonic exact order", xlabel="h", ylabel="Order p")
    axis.grid(True, alpha=0.3)
    order_path = _save(figure, directory / "heat_smooth_exact_order.png")
    return error_path, order_path


def _save(figure: plt.Figure, path: Path) -> Path:
    try:
        figure.tight_layout()
        figure.savefig(path, dpi=180)
    finally:
        plt.close(figure)
    return path
