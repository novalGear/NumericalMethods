from .convergence import (
    GridConvergenceStudy,
    restrict_to_coarse,
    run_grid_convergence,
    run_smooth_harmonic_verification,
)
from .experiments import (
    MethodComparison,
    SorScan,
    TimedSolverRun,
    compare_methods,
    scan_sor,
)
from .model import (
    BoundaryConditions,
    HeatProblem,
    SolverResult,
    SolverStatus,
    StoppingCriteria,
)
from .solvers import solve_gauss_seidel, solve_jacobi, solve_sor

__all__ = [
    "BoundaryConditions",
    "GridConvergenceStudy",
    "HeatProblem",
    "MethodComparison",
    "SolverResult",
    "SolverStatus",
    "SorScan",
    "StoppingCriteria",
    "TimedSolverRun",
    "compare_methods",
    "restrict_to_coarse",
    "run_grid_convergence",
    "run_smooth_harmonic_verification",
    "scan_sor",
    "solve_gauss_seidel",
    "solve_jacobi",
    "solve_sor",
]
