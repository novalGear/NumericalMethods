# Two-Dimensional Steady Heat Equation

This package solves the Laplace equation on a uniform rectangular grid with
independent Dirichlet data on all four sides. Jacobi, Gauss-Seidel, SOR,
matrix-free Conjugate Gradient (CG), and matrix-free BiCGSTAB are implemented
directly; no library PDE solver is used.

## Run

```bash
python -m pip install -r HeatEquation/requirements.txt
python -m HeatEquation.main
```

Use `python -m HeatEquation.main --help` for domain, grid, boundary,
stopping, SOR, and output options. Defaults use an `11 x 9` ordinary grid,
ten SOR values, three ordinary nested-grid levels, a five-level smooth exact
verification, and a four-grid solver-scaling study.

## Public Model

`HeatProblem` contains only physical dimensions, grid dimensions, and
boundary data. `StoppingCriteria` independently controls tolerance, iteration
limit, and a validated `StoppingRule`. The default rule is
`absolute_max_residual`, preserving the original CLI behavior. Experiments may
instead select `relative_l2_residual`. Models are frozen and validated. Side
values may be constants or one-coordinate callables.

Fields have shape `(ny, nx)`, with rows corresponding to `y`. Rectangular
cells with `hx != hy` use the weighted five-point stencil. Each corner is the
overflow-safe average of its two adjacent side values.

The numerical solvers return immutable `SolverResult` objects with a
`SolverStatus` of `converged`, `max_iter`, `nonfinite`, or `breakdown`. They do
not measure time. `compare_methods` and `scan_sor` own timing through
experiment-level records. Method comparison rejects incomplete runs; an SOR scan retains
status-marked scalar points, excludes failed runs from best-omega selection,
and does not retain every solver history.

History starts at iteration zero. Its initial delta and relative delta are
`None`. The `relative_residual_l2` diagnostic is the true interior PDE
residual norm `||r_k||_2 / ||r_0||_2`; it starts at one, or zero for an exact
initial field. Every solver uses the selected shared stopping rule; changing
the rule does not change the diagnostics. Final fields are defensively copied
and read-only, histories contain no field snapshots, and boundary cells are
never updated.

`solve_cg` and `solve_bicgstab` solve the interior Dirichlet system using the
negative discrete Laplacian. Their operator is matrix-free and supports
unequal grid spacing. Both report true PDE residuals and explicit numerical
breakdown or nonfinite statuses.

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
discontinuities. Its default grids are `5`, `9`, `17`, `33`, and `65` nodes
per side, solved by matrix-free CG with a strict absolute residual tolerance.

`run_solver_scaling` times Jacobi, Gauss-Seidel, SOR, CG, and BiCGSTAB from
equivalent initial fields on `5 x 5`, `9 x 9`, `17 x 17`, and `33 x 33` smooth
harmonic grids. All runs share the default relative-L2 tolerance `1e-8`; SOR
uses the fixed documented default `omega=1.7`. The reported size is the number
of interior unknowns `N=(nx-2)(ny-2)`. The criterion is included in scaling CSV
and JSON output together with both the relative L2 residual used for stopping
and the final absolute maximum residual used by the original assignment.

## Outputs

The CLI writes method/history/SOR tables, scaling and convergence CSV files, a
JSON summary, and an NPZ field archive. PNG figures include individual and
combined relative-L2 residual histories, combined absolute and relative
update histories, SOR heatmap/contours/sections, the status-aware SOR scan,
solver iterations/runtime versus interior unknowns, ordinary grid error/order,
smooth exact error/order with second-order references, and consistently scaled
field and contour images for every ordinary nested-grid level under
`results/grid_fields/ordinary/`.

## Layout

- `model.py`: immutable input, stopping, status, and result models
- `discrete_problem.py`: boundaries, matrix-free operator, and residual evaluation
- `solvers.py`: stationary, CG, and BiCGSTAB numerical kernels
- `experiments.py`: timing, method comparison, SOR scan, and solver scaling
- `convergence.py`: ordinary nested-grid study and smooth verification
- `serialization.py`, `plotting.py`: result consumers
- `main.py`: CLI composition root

Run tests with `python -m unittest tests.test_heat_equation`.
