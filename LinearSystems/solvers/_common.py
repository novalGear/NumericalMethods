"""Shared numerical safeguards and iteration state management."""

from dataclasses import dataclass

import numpy as np

from ..model import (
    FloatMatrix,
    FloatVector,
    IterationPoint,
    LinearSystem,
    SolverResult,
    SolverStatus,
    StoppingCriteria,
)


def criteria_or_default(criteria: StoppingCriteria | None) -> StoppingCriteria:
    return StoppingCriteria() if criteria is None else criteria


def initial_vector(system: LinearSystem, x0: FloatVector | None) -> FloatVector:
    if x0 is None:
        return np.zeros(system.size, dtype=np.float64)
    vector = np.array(x0, dtype=np.float64, copy=True)
    if vector.ndim != 1 or vector.shape != system.rhs.shape:
        raise ValueError(f"x0 must have shape {system.rhs.shape}")
    return vector


def is_spd(system: LinearSystem) -> bool:
    return system.is_spd


def dangerous_denominator(value: float, scale: float) -> bool:
    if not np.isfinite(value) or not np.isfinite(scale):
        return True
    if scale == 0.0:
        return value == 0.0
    return abs(value) <= np.finfo(np.float64).eps * abs(scale)


def unavailable_result(method: str, reason: str, *, omega: float | None = None) -> SolverResult:
    return SolverResult(
        method=method,
        solution=None,
        status=SolverStatus.NOT_APPLICABLE,
        iterations=None,
        reason=reason,
        omega=omega,
    )


@dataclass(frozen=True)
class ResidualEvaluation:
    solution: FloatVector
    residual: FloatVector
    residual_abs: float
    residual_rel: float


class IterationMonitor:
    """Own true-residual diagnostics and atomically commit finite iterates."""

    def __init__(
        self,
        method: str,
        system: LinearSystem,
        criteria: StoppingCriteria,
        x0: FloatVector,
        *,
        omega: float | None = None,
    ) -> None:
        self.method = method
        self.matrix = system.matrix
        self.rhs = system.rhs
        self.criteria = criteria
        self.omega = omega
        self.rhs_norm = float(np.linalg.norm(self.rhs))
        self.threshold = criteria.atol + criteria.rtol * self.rhs_norm
        self.matvec_count = 0
        self.diagnostic_matvec_count = 0
        self.history: list[IterationPoint] = []
        self.solution: FloatVector | None = None
        self.residual: FloatVector | None = None
        self.initial_status: SolverStatus | None = None
        self.initial_reason: str | None = None

        if not np.all(np.isfinite(x0)):
            self.initial_status = SolverStatus.NONFINITE
            self.initial_reason = "initial approximation contains NaN or Inf"
            return
        evaluation = self.evaluate(x0)
        if evaluation is None:
            self.initial_status = SolverStatus.NONFINITE
            self.initial_reason = "initial true residual contains NaN or Inf"
            return
        self._commit_evaluation(0, evaluation)
        if evaluation.residual_abs <= self.threshold:
            self.initial_status = SolverStatus.CONVERGED

    @property
    def converged(self) -> bool:
        return self.initial_status is SolverStatus.CONVERGED

    def matvec(
        self,
        vector: FloatVector,
        operator: FloatMatrix | None = None,
    ) -> FloatVector:
        self.matvec_count += 1
        with np.errstate(over="ignore", invalid="ignore"):
            return (self.matrix if operator is None else operator) @ vector

    def evaluate(self, solution: FloatVector) -> ResidualEvaluation | None:
        candidate = np.array(solution, dtype=np.float64, copy=True)
        if candidate.shape != self.rhs.shape or not np.all(np.isfinite(candidate)):
            return None
        self.diagnostic_matvec_count += 1
        with np.errstate(over="ignore", invalid="ignore"):
            residual = self.rhs - self.matrix @ candidate
            residual_abs = float(np.linalg.norm(residual))
        if not np.all(np.isfinite(residual)) or not np.isfinite(residual_abs):
            return None
        residual_rel = residual_abs / self.rhs_norm if self.rhs_norm != 0.0 else residual_abs
        return ResidualEvaluation(candidate, residual, residual_abs, residual_rel)

    def commit(
        self,
        iteration: int,
        solution: FloatVector,
        *,
        only_if_converged: bool = False,
    ) -> SolverStatus | None:
        evaluation = self.evaluate(solution)
        if evaluation is None:
            return SolverStatus.NONFINITE
        converged = evaluation.residual_abs <= self.threshold
        if not only_if_converged or converged:
            self._commit_evaluation(iteration, evaluation)
        return SolverStatus.CONVERGED if converged else None

    def _commit_evaluation(self, iteration: int, evaluation: ResidualEvaluation) -> None:
        self.solution = evaluation.solution
        self.residual = evaluation.residual
        self.history.append(
            IterationPoint(
                iteration=iteration,
                solution=evaluation.solution,
                residual_abs=evaluation.residual_abs,
                residual_rel=evaluation.residual_rel,
                matvec_count=self.matvec_count,
                diagnostic_matvec_count=self.diagnostic_matvec_count,
            )
        )

    def result(self, status: SolverStatus, reason: str | None = None) -> SolverResult:
        iterations = self.history[-1].iteration if self.history else 0
        return SolverResult(
            method=self.method,
            solution=self.solution,
            status=status,
            iterations=iterations,
            history=tuple(self.history),
            matvec_count=self.matvec_count,
            diagnostic_matvec_count=self.diagnostic_matvec_count,
            reason=reason,
            omega=self.omega,
        )

    def initial_result(self) -> SolverResult | None:
        if self.initial_status is None:
            return None
        return self.result(self.initial_status, self.initial_reason)

    def max_iter_result(self) -> SolverResult:
        return self.result(
            SolverStatus.MAX_ITER,
            f"maximum iteration count ({self.criteria.max_iter}) reached",
        )
