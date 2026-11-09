import argparse
from pathlib import Path

import numpy as np

from .convergence import (
    GridConvergenceStudy,
    run_grid_convergence,
    run_smooth_harmonic_verification,
)
from .experiments import compare_methods, scan_sor
from .model import BoundaryConditions, HeatProblem, StoppingCriteria
from .plotting import save_experiment_plots
from .serialization import write_experiment


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the steady two-dimensional heat equation."
    )
    parser.add_argument("--nx", type=int, default=11)
    parser.add_argument("--ny", type=int, default=9)
    parser.add_argument("--lx", type=float, default=2.0)
    parser.add_argument("--ly", type=float, default=1.0)
    parser.add_argument("--left", type=float, default=0.0)
    parser.add_argument("--right", type=float, default=100.0)
    parser.add_argument("--bottom", type=float, default=20.0)
    parser.add_argument("--top", type=float, default=60.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=100_000)
    parser.add_argument("--omega", type=float, default=1.7)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    problem = HeatProblem(
        args.lx,
        args.ly,
        args.nx,
        args.ny,
        BoundaryConditions(args.left, args.right, args.bottom, args.top),
    )
    criteria = StoppingCriteria(args.tolerance, args.max_iterations)
    comparison = compare_methods(problem, criteria, args.omega)
    scan = scan_sor(
        problem,
        tuple(float(value) for value in np.arange(1.0, 2.0, 0.1)),
        criteria,
    )
    ordinary = run_grid_convergence(
        problem,
        StoppingCriteria(1e-9, max(args.max_iterations, 200_000)),
        levels=3,
        omega=args.omega,
    )
    smooth = run_smooth_harmonic_verification(levels=4)
    output_paths = (
        *write_experiment(
            comparison, scan, args.output_dir, ordinary, smooth
        ),
        *save_experiment_plots(
            comparison, scan, args.output_dir, ordinary, smooth
        ),
    )

    print(
        f"{'Method':<16} {'Status':<10} {'Iterations':>10} "
        f"{'Residual':>13} {'Runtime (s)':>12}"
    )
    for run in comparison.runs:
        result = run.result
        print(
            f"{result.method:<16} {result.status.value:<10} "
            f"{result.iterations:>10d} {result.final_residual:>13.3e} "
            f"{run.elapsed_seconds:>12.4f}"
        )

    print(f"\n{'omega':>7} {'Status':<10} {'Iterations':>10} {'Residual':>13}")
    for point in scan.points:
        print(
            f"{point.omega:>7.2f} {point.status.value:<10} "
            f"{point.iterations:>10d} {point.final_residual:>13.3e}"
        )
    best = "none" if scan.best_omega is None else f"{scan.best_omega:g}"
    print(f"Best converged SOR omega: {best}")

    _print_study("Ordinary nested-grid study", ordinary, exact=False)
    _print_study("Smooth harmonic exact verification", smooth, exact=True)
    print(f"\nWrote {len(output_paths)} files to {args.output_dir}")


def _print_study(
    title: str, study: GridConvergenceStudy, *, exact: bool
) -> None:
    error_label = "Exact error" if exact else "E(h)"
    print(
        f"\n{title}\n"
        f"{'Grid':<12} {'Status':<10} {'h':>10} {'Iterations':>10} "
        f"{'Residual':>13} {error_label:>13} {'p':>8}"
    )
    for point in study.points:
        error = point.exact_error if exact else point.grid_difference
        order = point.exact_order if exact else point.observed_order
        error_text = "-" if error is None else f"{error:.3e}"
        order_text = "-" if order is None else f"{order:.3f}"
        grid = f"{point.grid[0]}x{point.grid[1]}"
        print(
            f"{grid:<12} {point.status.value:<10} {point.h:>10.3e} "
            f"{point.iterations:>10d} {point.final_residual:>13.3e} "
            f"{error_text:>13} {order_text:>8}"
        )


if __name__ == "__main__":
    main()
