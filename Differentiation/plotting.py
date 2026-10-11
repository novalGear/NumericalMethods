from collections.abc import Sequence
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt

from .diff_engine import DifferentiationResult


DEFAULT_PLOT_DIR = Path(__file__).resolve().parent / "plots"
logger = logging.getLogger(__name__)


def save_error_plots(
    results: Sequence[DifferentiationResult],
    output_dir: Path | str = DEFAULT_PLOT_DIR,
) -> tuple[Path, ...]:
    """Save one log-log error figure per function in PNG and PDF formats."""
    if not results:
        raise ValueError("results must not be empty")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    results_by_function: dict[str, list[DifferentiationResult]] = {}
    for result in results:
        results_by_function.setdefault(result.function_name, []).append(result)

    saved_paths: list[Path] = []
    for function_index, (function_name, function_results) in enumerate(
        results_by_function.items(),
        start=1,
    ):
        x_values = {result.x for result in function_results}
        if len(x_values) != 1:
            raise ValueError(
                f"function [{function_name}] has results for multiple x values"
            )
        x = x_values.pop()

        results_by_method: dict[str, list[DifferentiationResult]] = {}
        for result in function_results:
            results_by_method.setdefault(result.method_name, []).append(result)

        figure, axis = plt.subplots(figsize=(8, 5.5))
        omitted_zero_count = 0
        try:
            for method_name, method_results in results_by_method.items():
                ordered_results = sorted(method_results, key=lambda result: result.h)
                if any(
                    not math.isfinite(result.abs_error)
                    for result in ordered_results
                ):
                    raise ValueError(
                        f"method [{method_name}] for function [{function_name}] "
                        "contains a non-finite error"
                    )
                positive_results = [
                    result for result in ordered_results if result.abs_error > 0.0
                ]
                omitted_zero_count += len(ordered_results) - len(positive_results)

                axis.plot(
                    [result.h for result in positive_results],
                    [result.abs_error for result in positive_results],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.2,
                    label=method_name,
                )

            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("Step h")
            axis.set_ylabel("Absolute error")
            axis.set_title(f"Differentiation error for {function_name} at x = {x:g}")
            axis.grid(which="both", linestyle=":", alpha=0.6)
            axis.legend()

            if omitted_zero_count:
                logger.info(
                    "Omitted %d zero-error points from log plot for [%s]",
                    omitted_zero_count,
                    function_name,
                )
                axis.text(
                    0.01,
                    0.01,
                    f"Zero-error points omitted: {omitted_zero_count}",
                    transform=axis.transAxes,
                    fontsize=8,
                    verticalalignment="bottom",
                )

            figure.tight_layout()
            file_stem = f"diff_f{function_index:02d}"
            for suffix in ("png", "pdf"):
                output_path = destination / f"{file_stem}.{suffix}"
                figure.savefig(output_path, dpi=300, bbox_inches="tight")
                saved_paths.append(output_path)
        finally:
            plt.close(figure)

    return tuple(saved_paths)
