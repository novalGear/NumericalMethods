"""Conjugate-gradient Krylov solvers."""

import numpy as np

from ..model import FloatVector, LinearSystem, SolverResult, SolverStatus, StoppingCriteria
from ._common import (
    IterationMonitor,
    criteria_or_default,
    dangerous_denominator,
    initial_vector,
    is_spd,
    unavailable_result,
)


def solve_cg(
    system: LinearSystem,
    x0: FloatVector | None = None,
    criteria: StoppingCriteria | None = None,
) -> SolverResult:
    """Solve an SPD system by the conjugate-gradient method."""
    if not is_spd(system):
        return unavailable_result("cg", "conjugate gradients requires an SPD matrix")
    monitor = IterationMonitor("cg", system, criteria_or_default(criteria), initial_vector(system, x0))
    initial = monitor.initial_result()
    if initial is not None:
        return initial
    assert monitor.solution is not None and monitor.residual is not None
    solution = monitor.solution.copy()
    residual = monitor.residual.copy()
    direction = residual.copy()
    residual_squared = float(residual @ residual)

    for iteration in range(1, monitor.criteria.max_iter + 1):
        image = monitor.matvec(direction)
        if not np.all(np.isfinite(image)):
            return monitor.result(SolverStatus.NONFINITE, "A times search direction contains NaN or Inf")
        denominator = float(direction @ image)
        scale = float(np.linalg.norm(direction)) * float(np.linalg.norm(image))
        if not np.isfinite(denominator) or not np.isfinite(scale):
            return monitor.result(SolverStatus.NONFINITE, "CG step scalar is nonfinite")
        if dangerous_denominator(denominator, scale):
            return monitor.result(SolverStatus.BREAKDOWN, "CG step denominator is zero or numerically dangerous")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            alpha = residual_squared / denominator
            candidate = solution + alpha * direction
            next_residual = residual - alpha * image
        status = monitor.commit(iteration, candidate)
        if status is not None:
            reason = "CG iterate contains NaN or Inf" if status is SolverStatus.NONFINITE else None
            return monitor.result(status, reason)
        next_squared = float(next_residual @ next_residual)
        if not np.isfinite(next_squared) or not np.isfinite(alpha):
            return monitor.result(SolverStatus.NONFINITE, "CG recurrence contains NaN or Inf")
        if dangerous_denominator(residual_squared, float(np.linalg.norm(residual)) ** 2):
            return monitor.result(SolverStatus.BREAKDOWN, "CG beta denominator is zero or numerically dangerous")
        beta = next_squared / residual_squared
        direction = next_residual + beta * direction
        solution = candidate
        residual = next_residual
        residual_squared = next_squared
    return monitor.max_iter_result()


def solve_bicgstab(
    system: LinearSystem,
    x0: FloatVector | None = None,
    criteria: StoppingCriteria | None = None,
) -> SolverResult:
    """Solve a general square system using BiCGStab."""
    monitor = IterationMonitor(
        "bicgstab",
        system,
        criteria_or_default(criteria),
        initial_vector(system, x0),
    )
    initial = monitor.initial_result()
    if initial is not None:
        return initial
    assert monitor.solution is not None and monitor.residual is not None
    solution = monitor.solution.copy()
    residual = monitor.residual.copy()
    shadow = residual.copy()
    direction = residual.copy()

    for iteration in range(1, monitor.criteria.max_iter + 1):
        rho = float(shadow @ residual)
        rho_scale = float(np.linalg.norm(shadow)) * float(np.linalg.norm(residual))
        if not np.isfinite(rho) or not np.isfinite(rho_scale):
            return monitor.result(SolverStatus.NONFINITE, "BiCGStab rho is nonfinite")
        if dangerous_denominator(rho, rho_scale):
            return monitor.result(SolverStatus.BREAKDOWN, "BiCGStab rho is zero or numerically dangerous")
        image = monitor.matvec(direction)
        if not np.all(np.isfinite(image)):
            return monitor.result(SolverStatus.NONFINITE, "A times search direction contains NaN or Inf")
        alpha_denominator = float(shadow @ image)
        alpha_scale = float(np.linalg.norm(shadow)) * float(np.linalg.norm(image))
        if not np.isfinite(alpha_denominator) or not np.isfinite(alpha_scale):
            return monitor.result(SolverStatus.NONFINITE, "BiCGStab alpha scalar is nonfinite")
        if dangerous_denominator(alpha_denominator, alpha_scale):
            return monitor.result(SolverStatus.BREAKDOWN, "BiCGStab alpha denominator is dangerous")
        alpha = rho / alpha_denominator
        intermediate_residual = residual - alpha * image
        intermediate_solution = solution + alpha * direction
        status = monitor.commit(iteration, intermediate_solution, only_if_converged=True)
        if status is not None:
            reason = "BiCGStab intermediate iterate contains NaN or Inf" if status is SolverStatus.NONFINITE else None
            return monitor.result(status, reason)

        image_s = monitor.matvec(intermediate_residual)
        if not np.all(np.isfinite(image_s)):
            return monitor.result(SolverStatus.NONFINITE, "A times intermediate residual contains NaN or Inf")
        omega_denominator = float(image_s @ image_s)
        omega_scale = float(np.linalg.norm(image_s)) ** 2
        if not np.isfinite(omega_denominator) or not np.isfinite(omega_scale):
            return monitor.result(SolverStatus.NONFINITE, "BiCGStab omega scalar is nonfinite")
        if dangerous_denominator(omega_denominator, omega_scale):
            return monitor.result(SolverStatus.BREAKDOWN, "BiCGStab omega denominator is dangerous")
        omega_numerator = float(image_s @ intermediate_residual)
        omega_scale = float(np.linalg.norm(image_s)) * float(
            np.linalg.norm(intermediate_residual)
        )
        if not np.isfinite(omega_numerator) or not np.isfinite(omega_scale):
            return monitor.result(SolverStatus.NONFINITE, "BiCGStab omega scalar is nonfinite")
        if dangerous_denominator(omega_numerator, omega_scale):
            return monitor.result(
                SolverStatus.BREAKDOWN,
                "BiCGStab omega numerator is zero or numerically dangerous",
            )
        omega = omega_numerator / omega_denominator
        candidate = intermediate_solution + omega * intermediate_residual
        next_residual = intermediate_residual - omega * image_s
        status = monitor.commit(iteration, candidate)
        if status is not None:
            reason = "BiCGStab iterate contains NaN or Inf" if status is SolverStatus.NONFINITE else None
            return monitor.result(status, reason)

        next_rho = float(shadow @ next_residual)
        next_rho_scale = float(np.linalg.norm(shadow)) * float(np.linalg.norm(next_residual))
        if not np.isfinite(next_rho) or not np.isfinite(next_rho_scale):
            return monitor.result(SolverStatus.NONFINITE, "BiCGStab next rho is nonfinite")
        if dangerous_denominator(next_rho, next_rho_scale):
            return monitor.result(SolverStatus.BREAKDOWN, "BiCGStab next rho is zero or numerically dangerous")
        beta = (next_rho / rho) * (alpha / omega)
        with np.errstate(over="ignore", invalid="ignore"):
            direction = next_residual + beta * (direction - omega * image)
        if not np.all(np.isfinite(direction)) or not np.isfinite(beta):
            return monitor.result(SolverStatus.NONFINITE, "BiCGStab recurrence contains NaN or Inf")
        solution = candidate
        residual = next_residual
    return monitor.max_iter_result()
