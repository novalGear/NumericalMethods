import numpy as np

from .model import LinearSystem, MatrixDiagnostics


def build_variant_l(size: int = 20) -> LinearSystem:
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size <= 0:
        raise ValueError("size must be positive")

    indices = np.arange(1, size + 1, dtype=np.float64)
    matrix = 1.0 / (indices[:, None] + indices[None, :])
    np.fill_diagonal(matrix, 10.0)
    rhs = 1.0 / indices
    return LinearSystem(matrix=matrix, rhs=rhs, variant="l")


def diagnose_system(system: LinearSystem) -> MatrixDiagnostics:
    matrix = system.matrix
    diagonal = np.abs(np.diag(matrix))
    off_diagonal_sums = np.sum(np.abs(matrix), axis=1) - diagonal
    margin = float(np.min(diagonal - off_diagonal_sums))
    symmetric = bool(np.allclose(matrix, matrix.T))

    if symmetric:
        eigenvalues = np.linalg.eigvalsh(matrix)
    else:
        eigenvalues = np.linalg.eigvals(matrix)
        if np.any(np.abs(eigenvalues.imag) > 1e-12):
            eigenvalue_min = float("nan")
            eigenvalue_max = float("nan")
            return MatrixDiagnostics(
                size=system.size,
                symmetric=False,
                diagonal_dominance_margin=margin,
                eigenvalue_min=eigenvalue_min,
                eigenvalue_max=eigenvalue_max,
                condition_number_2=float(np.linalg.cond(matrix, 2)),
            )
        eigenvalues = eigenvalues.real

    return MatrixDiagnostics(
        size=system.size,
        symmetric=symmetric,
        diagonal_dominance_margin=margin,
        eigenvalue_min=float(np.min(eigenvalues)),
        eigenvalue_max=float(np.max(eigenvalues)),
        condition_number_2=float(np.linalg.cond(matrix, 2)),
    )
