import numpy as np

from .model import FloatArray, HeatProblem, readonly_array


def apply_boundary_conditions(problem: HeatProblem) -> FloatArray:
    """Create a field with fixed Dirichlet data and adjacent-side corner averages."""
    field = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    left = problem.boundaries.evaluate("left", problem.y)
    right = problem.boundaries.evaluate("right", problem.y)
    bottom = problem.boundaries.evaluate("bottom", problem.x)
    top = problem.boundaries.evaluate("top", problem.x)
    field[:, 0], field[:, -1] = left, right
    field[0, :], field[-1, :] = bottom, top
    # Weighted addition avoids overflowing when both adjacent values are large.
    field[0, 0] = 0.5 * left[0] + 0.5 * bottom[0]
    field[0, -1] = 0.5 * right[0] + 0.5 * bottom[-1]
    field[-1, 0] = 0.5 * left[-1] + 0.5 * top[0]
    field[-1, -1] = 0.5 * right[-1] + 0.5 * top[-1]
    if not np.all(np.isfinite(field)):
        raise ValueError("boundary construction produced a non-finite corner value")
    return readonly_array(field, ndim=2)


def initial_field(problem: HeatProblem) -> FloatArray:
    field = np.array(apply_boundary_conditions(problem), copy=True)
    boundary_values = np.concatenate((field[0], field[-1], field[1:-1, 0], field[1:-1, -1]))
    scale = float(np.max(np.abs(boundary_values)))
    mean = 0.0 if scale == 0.0 else scale * float(np.mean(boundary_values / scale))
    if not np.isfinite(mean):
        raise ValueError("boundary values do not have a finite initial mean")
    field[1:-1, 1:-1] = mean
    return field


def apply_interior_operator(problem: HeatProblem, interior: FloatArray) -> FloatArray:
    """Apply the negative discrete Laplacian with zero exterior values."""
    values = np.asarray(interior, dtype=np.float64)
    expected = (problem.ny - 2, problem.nx - 2)
    if values.shape != expected:
        raise ValueError(f"interior must have shape {expected}")
    zero_extended = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    zero_extended[1:-1, 1:-1] = values
    return _negative_laplacian(problem, zero_extended)


def interior_residual(problem: HeatProblem, field: FloatArray) -> FloatArray:
    """Return b - A*u for the interior Dirichlet system."""
    if field.shape != (problem.ny, problem.nx):
        raise ValueError(f"field must have shape {(problem.ny, problem.nx)}")
    return -_negative_laplacian(problem, field)


def _negative_laplacian(problem: HeatProblem, field: FloatArray) -> FloatArray:
    interior = field[1:-1, 1:-1]
    with np.errstate(over="ignore", invalid="ignore"):
        difference_x = (
            interior - field[1:-1, :-2] + interior - field[1:-1, 2:]
        )
        difference_y = (
            interior - field[:-2, 1:-1] + interior - field[2:, 1:-1]
        )
        return difference_x / problem.hx**2 + difference_y / problem.hy**2


def residual_max(problem: HeatProblem, field: FloatArray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        maximum = np.max(np.abs(interior_residual(problem, field)))
    return float(maximum)


def residual_l2(problem: HeatProblem, field: FloatArray) -> float:
    """Return the Euclidean norm of the true interior PDE residual."""
    residual = interior_residual(problem, field)
    scale = float(np.max(np.abs(residual)))
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        return float(scale * np.linalg.norm(residual / scale))
