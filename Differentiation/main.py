import logging
import math

logger = logging.getLogger()

from .diff_engine import DifferentiationEngine
from .diff_methods import differentiation_methods

differentiator: DifferentiationEngine = DifferentiationEngine()

for method_name, method in differentiation_methods.items():
    differentiator.add_method(method_name, method)

differentiator.add_sample(
    "sin(x^2)",
    lambda x: math.sin(x**2),
    lambda x: 2 * x * math.cos(x**2),
)
differentiator.add_sample(
    "cos(sin(x))",
    lambda x: math.cos(math.sin(x)),
    lambda x: -math.sin(math.sin(x)) * math.cos(x),
)
differentiator.add_sample(
    "exp(sin(cos(x)))",
    lambda x: math.exp(math.sin(math.cos(x))),
    lambda x: (
        -math.exp(math.sin(math.cos(x)))
        * math.cos(math.cos(x))
        * math.sin(x)
    ),
)
differentiator.add_sample(
    "ln(x+3)",
    lambda x: math.log(x + 3),
    lambda x: 1 / (x + 3),
)
differentiator.add_sample(
    "(x+3)^0.5",
    lambda x: math.sqrt(x + 3),
    lambda x: 1 / (2 * math.sqrt(x + 3)),
)


def main() -> None:
    from .plotting import save_error_plots

    results = differentiator.run_all(x=1.0)
    for output_path in save_error_plots(results):
        print(output_path)


if __name__ == "__main__":
    main()
