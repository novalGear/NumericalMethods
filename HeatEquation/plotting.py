from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .convergence import GridConvergenceStudy
from .experiments import MethodComparison, SolverScalingStudy, SorScan
from .model import SolverResult, SolverStatus


def save_experiment_plots(
    comparison: MethodComparison,
    sor_scan: SorScan,
    output_dir: str | PathLike[str],
    grid_convergence: GridConvergenceStudy | None = None,
    smooth_verification: GridConvergenceStudy | None = None,
    solver_scaling: SolverScalingStudy | None = None,
) -> tuple[Path, ...]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for run in comparison.runs:
        result = run.result
        figure, axis = plt.subplots(figsize=(7, 4.5))
        _plot_positive_history(axis, result, "relative_residual_l2")
        axis.set(
            title=f"{result.method} relative L2 residual convergence",
            xlabel="Iteration",
            ylabel=r"Relative L2 residual $||r_k||_2 / ||r_0||_2$",
        )
        axis.grid(True, which="both", alpha=0.3)
        paths.append(
            _save(figure, directory / f"heat_residual_{result.method}.png")
        )

    paths.append(
        _save_history_comparison(
            comparison,
            directory / "heat_residual_all.png",
            "relative_residual_l2",
            "Iterative-method relative L2 residual convergence",
            r"Relative L2 residual $||r_k||_2 / ||r_0||_2$",
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

    result = next(
        run.result for run in comparison.runs if run.result.method == "sor"
    )
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
    if solver_scaling is not None:
        paths.extend(_save_scaling_plots(solver_scaling, directory))
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
    return error_path, order_path, *_save_grid_field_plots(study, directory)


def _save_grid_field_plots(
    study: GridConvergenceStudy, directory: Path
) -> tuple[Path, ...]:
    series_directory = directory / "grid_fields" / "ordinary"
    if series_directory.is_dir():
        for pattern in ("heat_field_level*.png", "heat_contours_level*.png"):
            for stale_path in series_directory.glob(pattern):
                stale_path.unlink()
    if not study.snapshots:
        return ()
    series_directory.mkdir(parents=True, exist_ok=True)

    raw_min = min(float(snapshot.field.min()) for snapshot in study.snapshots)
    raw_max = max(float(snapshot.field.max()) for snapshot in study.snapshots)
    if raw_min == raw_max:
        padding = max(1.0, abs(raw_min)) * 1e-6
        vmin, vmax = raw_min - padding, raw_max + padding
        contour_levels = np.array([], dtype=np.float64)
    else:
        vmin, vmax = raw_min, raw_max
        contour_levels = np.linspace(vmin, vmax, 14)[1:-1]

    paths: list[Path] = []
    for level, snapshot in enumerate(study.snapshots):
        problem = snapshot.problem
        suffix = f"level{level:02d}_nx{problem.nx}_ny{problem.ny}.png"
        details = (
            f"level {level}, {problem.nx} x {problem.ny}, "
            f"hx={problem.hx:.3g}, hy={problem.hy:.3g}"
        )
        figure, axis = plt.subplots(figsize=(7, 5))
        image = axis.pcolormesh(
            problem.x,
            problem.y,
            snapshot.field,
            shading="nearest",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        figure.colorbar(image, ax=axis, label="Temperature")
        axis.set(
            title=f"Temperature field ({details})",
            xlabel="x",
            ylabel="y",
            xlim=(0, problem.lx),
            ylim=(0, problem.ly),
        )
        paths.append(_save(figure, series_directory / f"heat_field_{suffix}"))

        figure, axis = plt.subplots(figsize=(7, 5))
        local_min = float(snapshot.field.min())
        local_max = float(snapshot.field.max())
        local_levels = contour_levels[
            (contour_levels > local_min) & (contour_levels < local_max)
        ]
        if local_levels.size:
            contours = axis.contour(
                problem.x,
                problem.y,
                snapshot.field,
                levels=local_levels,
            )
            axis.clabel(contours, inline=True, fontsize=8)
        elif local_min == local_max:
            axis.text(
                0.5,
                0.5,
                f"T = {local_min:g}",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No shared isotherms in this range",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.set(
            title=f"Temperature isotherms ({details})",
            xlabel="x",
            ylabel="y",
            xlim=(0, problem.lx),
            ylim=(0, problem.ly),
        )
        paths.append(_save(figure, series_directory / f"heat_contours_{suffix}"))

    return tuple(paths)


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


def _save_scaling_plots(
    study: SolverScalingStudy, directory: Path
) -> tuple[Path, ...]:
    output: list[Path] = []
    criterion = study.criteria.rule.value.replace("_", " ")
    for attribute, filename, title, ylabel in (
        (
            "iterations",
            "heat_iterations_vs_unknowns.png",
            "Solver iterations versus grid size",
            "Iterations",
        ),
        (
            "elapsed_seconds",
            "heat_runtime_vs_unknowns.png",
            "Solver runtime versus grid size",
            "Runtime (seconds)",
        ),
    ):
        figure, axis = plt.subplots(figsize=(7, 4.5))
        methods = tuple(dict.fromkeys(point.method for point in study.points))
        for method in methods:
            points = [point for point in study.points if point.method == method]
            axis.loglog(
                [point.unknowns for point in points],
                [getattr(point, attribute) for point in points],
                marker="o",
                label=method,
            )
        axis.set(
            title=f"{title}\n{criterion}, tolerance={study.criteria.tolerance:g}",
            xlabel="Interior unknowns N",
            ylabel=ylabel,
        )
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
        output.append(_save(figure, directory / filename))
    return tuple(output)


def _save(figure: plt.Figure, path: Path) -> Path:
    try:
        figure.tight_layout()
        figure.savefig(path, dpi=180)
    finally:
        plt.close(figure)
    return path
