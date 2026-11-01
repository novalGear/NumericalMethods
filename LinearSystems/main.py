import argparse
import logging
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment
from .plotting import save_experiment_plots
from .serialization import write_experiment
from .systems import build_variant_l


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the numerical linear-system experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = run_experiment(build_variant_l(), ExperimentConfig())
    output_paths = (
        *write_experiment(results, args.output_dir),
        *save_experiment_plots(results, args.output_dir),
    )

    for metric in results.metrics:
        iterations = "-" if metric.iterations is None else str(metric.iterations)
        print(
            f"{metric.method:20} status={metric.status.value:10} "
            f"iterations={iterations:>5} residual={metric.residual_rel:.3e}"
        )
    print(f"Best tested SOR omega(s): {results.best_sor_omegas}")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
