# Linear Systems

## Task

Solve the dense system $Ax=b$ for variant `l` with $N=20$:

$$
a_{ii}=10, \qquad a_{ij}=\frac{1}{i+j}\ (i\ne j), \qquad b_i=\frac{1}{i},
$$

where mathematical indices start at one. The experiment implements Gaussian
elimination with partial pivoting, Doolittle LU, Jacobi, Gauss-Seidel, SOR,
steepest descent, one-step minimal residual, CG, and BiCGStab.

Matrices use two-dimensional NumPy `float64` arrays with shape `(N, N)`.
Vectors use one-dimensional arrays with shape `(N,)`. `LinearSystem` validates
and stores read-only copies so that solvers cannot mutate shared experiment
inputs.

## Sources

- `model.py` defines validated systems, stopping criteria, statuses, histories,
  metrics, and experiment result data.
- `systems.py` generates variant `l` and calculates matrix diagnostics.
- `solvers/` contains the nine numerical algorithms and shared true-residual
  monitoring. Solver-specific update equations remain in their family module.
- `experiment.py` scans SOR parameters, benchmarks methods, selects the best
  tested over-relaxation parameter, and calculates common metrics.
- `serialization.py` writes the required CSV, JSON, and NPZ files.
- `plotting.py` writes residual-history and SOR-scan figures.
- `main.py` is the composition root and command-line entry point.

The numerical algorithms do not write files, plot, benchmark themselves, or
use the library reference solution for convergence decisions.

## Launch

From the repository root, use the shared virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r LinearSystems/requirements.txt
.venv/bin/python -m LinearSystems.main
```

Use another output directory if needed:

```bash
.venv/bin/python -m LinearSystems.main --output-dir /tmp/linear-results
```

The default output directory is `LinearSystems/results/`.

## Outputs

The run creates:

- `slau_metrics.csv` with common metrics for all nine methods;
- `slau_history.csv` with true-residual histories for seven iterative methods;
- `slau_sor_scan.csv` with every tested relaxation parameter;
- `slau_solution.csv` with reference and computed solution vectors;
- `matrix_diagnostics.json` and `system.npz`;
- `slau_residual_vs_iteration.png` and `.pdf`;
- `slau_sor_iterations_vs_omega.png` and `.pdf`.

The default stopping condition is

$$
\|b-Ax_k\|_2 \le 10^{-14}+10^{-10}\|b\|_2.
$$

Histories begin at $k=0$. A `max_iter` status means only that the configured
iteration limit was reached; it is not reported as proof of divergence.

Run the numerical and integration checks with:

```bash
.venv/bin/python -m unittest tests.test_linear_systems
```
