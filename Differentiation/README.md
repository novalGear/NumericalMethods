# Numerical Differentiation

## Task

Approximate the first derivative of five functions at the fixed point

$$
x_0 = 1.
$$

The value $x_0=1$ is a working assumption because the original task does not
specify the evaluation point. It must remain configurable.

| Function | Analytical derivative |
| --- | --- |
| $f_1(x)=\sin(x^2)$ | $f_1'(x)=2x\cos(x^2)$ |
| $f_2(x)=\cos(\sin x)$ | $f_2'(x)=-\sin(\sin x)\cos x$ |
| $f_3(x)=\exp(\sin(\cos x))$ | $f_3'(x)=-\exp(\sin(\cos x))\cos(\cos x)\sin x$ |
| $f_4(x)=\ln(x+3)$ | $f_4'(x)=1/(x+3)$ |
| $f_5(x)=\sqrt{x+3}$ | $f_5'(x)=1/(2\sqrt{x+3})$ |

For every function and differentiation method, use the prescribed steps

$$
h_n=\frac{2}{2^n}=2^{1-n}, \qquad n=1,\ldots,21.
$$

Thus, the experiment covers steps from $h=1$ down to $h=2^{-20}$.

The absolute approximation error is

$$
E_j(h)=\left|D_j(h)-f'(x_0)\right|.
$$

## Differentiation Methods

### Forward difference

$$
D_1(h)=\frac{f(x_0+h)-f(x_0)}{h}.
$$

Its theoretical truncation-error order is $O(h)$.

### Backward difference

$$
D_2(h)=\frac{f(x_0)-f(x_0-h)}{h}.
$$

Its theoretical truncation-error order is $O(h)$.

### Central difference

$$
D_3(h)=\frac{f(x_0+h)-f(x_0-h)}{2h}.
$$

Symmetry eliminates the leading even-power terms in the numerator, giving an
error of $O(h^2)$.

### Fourth-order central formula

$$
D_4(h)=
\frac{4}{3}\frac{f(x_0+h)-f(x_0-h)}{2h}
-\frac{1}{3}\frac{f(x_0+2h)-f(x_0-2h)}{4h}.
$$

This linear combination cancels the $O(h^2)$ term of the central difference:

$$
D_4(h)=f'(x_0)+O(h^4).
$$

### Sixth-order central formula

$$
D_5(h)=
\frac{3}{2}\frac{f(x_0+h)-f(x_0-h)}{2h}
-\frac{3}{5}\frac{f(x_0+2h)-f(x_0-2h)}{4h}
+\frac{1}{10}\frac{f(x_0+3h)-f(x_0-3h)}{6h}.
$$

This combination cancels both the $O(h^2)$ and $O(h^4)$ terms:

$$
D_5(h)=f'(x_0)+O(h^6).
$$

## Expected Accuracy

In the truncation-dominated region, the expected ranking is:

1. sixth-order central formula;
2. fourth-order central formula;
3. second-order central difference;
4. forward and backward differences.

This ranking is not guaranteed for every tested value of $h$. The total
floating-point error can be described qualitatively by

$$
E(h)\approx C_T h^p+C_R\frac{\varepsilon_{\mathrm{mach}}}{h},
$$

where $p$ is the theoretical order and
$\varepsilon_{\mathrm{mach}}\approx2.22\cdot10^{-16}$ for `float64`.

For large $h$, truncation error dominates. Decreasing $h$ should then reduce
the error approximately as $h^p$. For very small $h$, subtraction of nearby
function values causes cancellation and round-off amplification. Each error
curve may therefore reach a minimum and then flatten or increase.

Higher-order formulas generally reach smaller errors with larger steps, but
they evaluate the function at more points and combine more floating-point
values. Consequently, they are not necessarily the most accurate methods at
every individual step.

Forward and backward differences have the same theoretical order, but their
error constants and signs differ. Their measured absolute errors therefore do
not need to coincide.

## Plots

The experiment produces exactly five primary figures:

- `diff_f01`: error curves for $\sin(x^2)$;
- `diff_f02`: error curves for $\cos(\sin x)$;
- `diff_f03`: error curves for $\exp(\sin(\cos x))$;
- `diff_f04`: error curves for $\ln(x+3)$;
- `diff_f05`: error curves for $\sqrt{x+3}$.

Each figure contains five curves, one for each differentiation formula. The
horizontal axis is the step $h$, and the vertical axis is the absolute error
$E_j(h)$. Both axes use logarithmic scales. Markers distinguish sampled steps,
and every figure includes a legend, grid, and function title. Figures are
saved in PNG and PDF formats.

The analytical derivative is the reference value and is not plotted as a
sixth curve.

If an error is exactly zero in `float64`, the zero remains in the numerical
data but is omitted from the logarithmic plot. It must not be replaced with an
arbitrary positive value. When this occurs, the figure contains the number of
omitted points and the command logs the same count for that function.

Plots are written to `Differentiation/plots/` with the following filenames:

```text
diff_f01.png  diff_f01.pdf
diff_f02.png  diff_f02.pdf
diff_f03.png  diff_f03.pdf
diff_f04.png  diff_f04.pdf
diff_f05.png  diff_f05.pdf
```

## Sources and Launch

The differentiation implementation is divided by responsibility:

- `diff_methods.py` contains the five pure finite-difference calculations;
- `diff_engine.py` configures and executes every function, method, and step
  combination and returns immutable result records;
- `main.py` declares the function samples and analytical derivatives and is the
  command-line entry point;
- `plotting.py` groups result records and saves the five required error plots;
- `requirements.txt` declares the plotting dependency.

From the repository root, create an isolated environment and install the
dependency with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r Differentiation/requirements.txt
```

Run the complete differentiation and plotting experiment with:

```bash
.venv/bin/python -m Differentiation.main
```

The command evaluates all five functions with all five methods for the default
inclusive range $n=1,\ldots,21$, prints the generated paths, and saves plots in
`Differentiation/plots/`.

## Observed Order

The observed convergence order can be estimated from adjacent steps by

$$
p_{\mathrm{obs}}=
\frac{\ln(E(h)/E(h/2))}{\ln 2}.
$$

Only points in the truncation-dominated region should be used. Zero errors and
points affected by round-off must be excluded from this estimate.

| Method | Expected order |
| --- | ---: |
| Forward difference | 1 |
| Backward difference | 1 |
| Central difference | 2 |
| Fourth-order central formula | 4 |
| Sixth-order central formula | 6 |

Special function values may suppress a leading error coefficient, causing the
measured order to differ from the general theoretical order.

## Output Data

The experiment should produce `diff_errors.csv` with one row for every
function, method, and tested step:

```text
function_id,x0,method,n,h,derivative_exact,derivative_numeric,abs_error
```

The expected number of data rows is
$5\times5\times21=525$.

The experiment should also produce `diff_summary.csv` with one summary row for
every function-method pair:

```text
function_id,method,theoretical_order,observed_order,slope_h_min,slope_h_max,h_best_tested,min_abs_error
```

The expected number of summary rows is $5\times5=25$.

## Experimental Results

This section must be completed after running the experiment. Generated results
should be referenced from:

- `results/lab1/diff_errors.csv`;
- `results/lab1/diff_summary.csv`;
- `Differentiation/plots/diff_f01.png` through `diff_f05.png`;
- the corresponding PDF figures.

| Function | Method | Theoretical order | Observed order | Best tested $h$ | Minimum absolute error |
| --- | --- | ---: | ---: | ---: | ---: |
| To be measured | To be measured | - | - | - | - |

The final analysis should address whether each method demonstrates its
theoretical order, which tested step minimizes its error, where round-off
begins to dominate, and whether higher-order methods are consistently more
accurate. Zero errors, overlapping curves, non-monotonic behavior, and other
deviations from the expectations are experimental results and must not be
hidden or replaced with theoretical values.
