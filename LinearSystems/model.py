from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


def readonly_vector(values: FloatVector) -> FloatVector:
    vector = np.array(values, dtype=np.float64, copy=True)
    if vector.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector must contain only finite values")
    vector.setflags(write=False)
    return vector


class SolverStatus(StrEnum):
    CONVERGED = "converged"
    MAX_ITER = "max_iter"
    BREAKDOWN = "breakdown"
    NOT_APPLICABLE = "not_applicable"
    NONFINITE = "nonfinite"


@dataclass(frozen=True)
class LinearSystem:
    matrix: FloatMatrix
    rhs: FloatVector
    variant: str

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=np.float64, copy=True)
        rhs = np.array(self.rhs, dtype=np.float64, copy=True)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        if rhs.ndim != 1 or rhs.shape[0] != matrix.shape[0]:
            raise ValueError("rhs must have shape (N,)")
        if matrix.shape[0] == 0:
            raise ValueError("system must not be empty")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
            raise ValueError("system must contain only finite values")

        matrix.setflags(write=False)
        rhs.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "rhs", rhs)

    @property
    def size(self) -> int:
        return self.rhs.size

    @cached_property
    def is_spd(self) -> bool:
        scale = float(np.max(np.abs(self.matrix)))
        if scale == 0.0:
            return False
        tolerance = 100.0 * np.finfo(np.float64).eps * scale
        if float(np.max(np.abs(self.matrix - self.matrix.T))) > tolerance:
            return False
        try:
            np.linalg.cholesky(self.matrix)
        except np.linalg.LinAlgError:
            return False
        return True


@dataclass(frozen=True)
class StoppingCriteria:
    rtol: float = 1e-10
    atol: float = 1e-14
    max_iter: int = 10_000

    def __post_init__(self) -> None:
        if not np.isfinite(self.rtol) or self.rtol < 0.0:
            raise ValueError("rtol must be finite and nonnegative")
        if not np.isfinite(self.atol) or self.atol < 0.0:
            raise ValueError("atol must be finite and nonnegative")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int):
            raise TypeError("max_iter must be an integer")
        if self.max_iter < 0:
            raise ValueError("max_iter must be nonnegative")


@dataclass(frozen=True)
class IterationPoint:
    iteration: int
    solution: FloatVector
    residual_abs: float
    residual_rel: float
    matvec_count: int
    diagnostic_matvec_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "solution", readonly_vector(self.solution))


@dataclass(frozen=True)
class SolverResult:
    method: str
    solution: FloatVector | None
    status: SolverStatus
    iterations: int | None
    history: tuple[IterationPoint, ...] = ()
    matvec_count: int = 0
    diagnostic_matvec_count: int = 0
    reason: str | None = None
    omega: float | None = None
    diagnostics: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.solution is not None:
            object.__setattr__(self, "solution", readonly_vector(self.solution))
        if self.iterations is not None and self.iterations < 0:
            raise ValueError("iterations must be nonnegative")


@dataclass(frozen=True)
class MatrixDiagnostics:
    size: int
    symmetric: bool
    diagonal_dominance_margin: float
    eigenvalue_min: float
    eigenvalue_max: float
    condition_number_2: float


@dataclass(frozen=True)
class SolverMetrics:
    method: str
    omega: float | None
    iterations: int | None
    matvec_count: int
    diagnostic_matvec_count: int
    elapsed_median_s: float
    residual_abs: float
    residual_rel: float
    backward_error: float
    reference_error_abs: float
    reference_error_rel: float
    status: SolverStatus


@dataclass(frozen=True)
class SorScanPoint:
    omega: float
    iterations: int | None
    elapsed_median_s: float
    residual_rel: float
    status: SolverStatus


@dataclass(frozen=True)
class ExperimentResults:
    system: LinearSystem
    diagnostics: MatrixDiagnostics
    reference_solution: FloatVector
    solver_results: tuple[SolverResult, ...]
    metrics: tuple[SolverMetrics, ...]
    sor_scan: tuple[SorScanPoint, ...]
    best_sor_omegas: tuple[float, ...]
    threshold_relative: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_solution",
            readonly_vector(self.reference_solution),
        )
