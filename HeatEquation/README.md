# Two-Dimensional Steady Heat Equation

This package solves the Laplace equation on a uniform rectangular grid with
independent Dirichlet data on all four sides. Jacobi, Gauss-Seidel, and SOR
are implemented directly; no library PDE solver is used.

## Run

```bash
python -m pip install -r HeatEquation/requirements.txt
python -m HeatEquation.main
```

Use `python -m HeatEquation.main --help` for domain, grid, boundary,
stopping, SOR, and output options. Defaults use an `11 x 9` ordinary grid,
ten SOR values, three ordinary nested-grid levels, and a separate four-level
smooth exact verification.

## Public Model

`HeatProblem` contains only physical dimensions, grid dimensions, and
boundary data. `StoppingCriteria` independently controls tolerance and the
iteration limit. Both are frozen and validated. Side values may be constants
or one-coordinate callables.

Fields have shape `(ny, nx)`, with rows corresponding to `y`. Rectangular
cells with `hx != hy` use the weighted five-point stencil. Each corner is the
overflow-safe average of its two adjacent side values.

The numerical solvers return immutable `SolverResult` objects with a
`SolverStatus` of `converged`, `max_iter`, or `nonfinite`. They do not measure
time. `compare_methods` and `scan_sor` own timing through experiment-level
records. Method comparison rejects incomplete runs; an SOR scan retains
status-marked scalar points, excludes failed runs from best-omega selection,
and does not retain every solver history.

History starts at iteration zero. Its initial delta and relative delta are
`None`, and an exact initial field returns zero iterations. The stopping rule
is the maximum absolute discrete residual; update diagnostics do not control
termination. Final fields are defensively copied and read-only, histories
contain no field snapshots, and boundary cells are never updated.

## Convergence Studies

`run_grid_convergence` studies the user problem on nested sizes
`2 * (N - 1) + 1`, restricts fine solutions to shared physical nodes, and
reports `E(h)` with independently derived observed order.

Different constant values on adjacent walls are discontinuous at corners, so
their maximum-norm ordinary-grid differences need not display second-order
decay on practical grids. That ordinary result is reported as-is rather than
mislabelled as a verification result.

`run_smooth_harmonic_verification` is deliberately separate. It uses the
non-polynomial harmonic function `exp(x) cos(y)` and reports both nested-grid
differences and true exact error, avoiding polynomial exactness and corner
discontinuities.

## Outputs

The CLI writes method/history/SOR tables, separately named ordinary and smooth
verification CSV files, a JSON summary, and an NPZ field archive. PNG figures
include individual and combined residual histories, combined absolute and
relative update histories, heatmap, contours, sections, status-aware SOR
scan, ordinary grid error/order, and smooth exact error/order.

## Layout

- `model.py`: immutable input, stopping, status, and result models
- `discrete_problem.py`: boundary construction and residual evaluation
- `solvers.py`: all stationary numerical kernels
- `experiments.py`: timing, method comparison, and streamed SOR scan
- `convergence.py`: ordinary nested-grid study and smooth verification
- `serialization.py`, `plotting.py`: result consumers
- `main.py`: CLI composition root

Run tests with `python -m unittest tests.test_heat_equation`.
