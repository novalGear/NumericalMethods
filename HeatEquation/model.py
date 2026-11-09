from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]
BoundaryValue: TypeAlias = float | Callable[[float], float]


def readonly_array(values: object, *, ndim: int | None = None) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"array must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("array must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class BoundaryConditions:
    left: BoundaryValue
    right: BoundaryValue
    bottom: BoundaryValue
    top: BoundaryValue

    def __post_init__(self) -> None:
        for name in ("left", "right", "bottom", "top"):
            value = getattr(self, name)
            if not callable(value):
                try:
                    finite = bool(np.isfinite(value))
                except TypeError as error:
                    raise ValueError(
                        f"{name} boundary must be finite or callable"
                    ) from error
                if not finite:
                    raise ValueError(f"{name} boundary must be finite or callable")

    def evaluate(self, side: str, coordinates: FloatArray) -> FloatArray:
        value = getattr(self, side)
        try:
            if callable(value):
                evaluated = np.array(
                    [value(float(coordinate)) for coordinate in coordinates],
                    dtype=np.float64,
                )
            else:
                evaluated = np.full(coordinates.shape, value, dtype=np.float64)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{side} boundary could not be evaluated") from error
        if not np.all(np.isfinite(evaluated)):
            raise ValueError(f"{side} boundary produced a non-finite value")
        return evaluated


@dataclass(frozen=True)
class HeatProblem:
    lx: float
    ly: float
    nx: int
    ny: int
    boundaries: BoundaryConditions

    def __post_init__(self) -> None:
        if not np.isfinite(self.lx) or self.lx <= 0:
            raise ValueError("lx must be finite and positive")
        if not np.isfinite(self.ly) or self.ly <= 0:
            raise ValueError("ly must be finite and positive")
        for name in ("nx", "ny"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 3:
                raise ValueError(f"{name} must be at least 3")
        if not isinstance(self.boundaries, BoundaryConditions):
            raise TypeError("boundaries must be BoundaryConditions")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ax = np.float64(1.0) / np.float64(self.hx) / np.float64(self.hx)
            ay = np.float64(1.0) / np.float64(self.hy) / np.float64(self.hy)
            denominator = np.float64(2.0) * (ax + ay)
        if not np.all(np.isfinite((self.hx, self.hy, ax, ay, denominator))):
            raise ValueError("grid spacing is too small for a finite-difference stencil")

    @property
    def hx(self) -> float:
        return self.lx / (self.nx - 1)

    @property
    def hy(self) -> float:
        return self.ly / (self.ny - 1)

    @property
    def x(self) -> FloatArray:
        return readonly_array(np.linspace(0.0, self.lx, self.nx), ndim=1)

    @property
    def y(self) -> FloatArray:
        return readonly_array(np.linspace(0.0, self.ly, self.ny), ndim=1)


@dataclass(frozen=True)
class IterationPoint:
    iteration: int
    residual: float
    delta: float | None
    relative_delta: float | None


@dataclass(frozen=True)
class StoppingCriteria:
    tolerance: float = 1e-8
    max_iterations: int = 100_000

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations, int
        ):
            raise TypeError("max_iterations must be an integer")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be nonnegative")


class SolverStatus(StrEnum):
    CONVERGED = "converged"
    MAX_ITER = "max_iter"
    NONFINITE = "nonfinite"


@dataclass(frozen=True)
class SolverResult:
    method: str
    field: FloatArray
    status: SolverStatus
    iterations: int
    final_residual: float
    history: tuple[IterationPoint, ...]
    omega: float | None = None

    def __post_init__(self) -> None:
        field = readonly_array(self.field, ndim=2)
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "history", tuple(self.history))
        if self.iterations < 0:
            raise ValueError("iterations must be nonnegative")

    @property
    def converged(self) -> bool:
        return self.status is SolverStatus.CONVERGED
