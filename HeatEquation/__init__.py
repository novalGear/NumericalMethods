from .convergence import (
    GridConvergenceStudy,
    restrict_to_coarse,
    run_grid_convergence,
    run_smooth_harmonic_verification,
)
from .experiments import (
    MethodComparison,
    SolverScalingPoint,
    SolverScalingStudy,
    SorScan,
    TimedSolverRun,
    compare_methods,
    run_solver_scaling,
    scan_sor,
)
from .model import (
    BoundaryConditions,
    HeatProblem,
    SolverResult,
    SolverStatus,
    StoppingCriteria,
    StoppingRule,
)
from .solvers import (
    solve_bicgstab,
    solve_cg,
    solve_gauss_seidel,
    solve_jacobi,
    solve_sor,
)

__all__ = [
    "BoundaryConditions",
    "GridConvergenceStudy",
    "HeatProblem",
    "MethodComparison",
    "SolverResult",
    "SolverScalingPoint",
    "SolverScalingStudy",
    "SolverStatus",
    "SorScan",
    "StoppingCriteria",
    "StoppingRule",
    "TimedSolverRun",
    "compare_methods",
    "restrict_to_coarse",
    "run_grid_convergence",
    "run_smooth_harmonic_verification",
    "run_solver_scaling",
    "scan_sor",
    "solve_bicgstab",
    "solve_cg",
    "solve_gauss_seidel",
    "solve_jacobi",
    "solve_sor",
]
