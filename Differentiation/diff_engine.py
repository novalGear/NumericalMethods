from collections.abc import Callable
from typing import Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

ScalarFunction = Callable[[float], float]
DifferentiationMethod = Callable[[ScalarFunction, float, float], float]

@dataclass(frozen=True)
class DifferentiationResult:
    function_name: str
    method_name: str
    x: float
    n: int
    h: float
    derivative_exact: float
    derivative_numeric: float
    abs_error: float

class DifferentiationEngine:
    _n_start: int
    _n_end:   int

    _function_samples:   Dict[str, ScalarFunction]
    _derivative_samples: Dict[str, ScalarFunction]

    _diff_methods: Dict[str, DifferentiationMethod]

    def __init__(self, n_start: int = 1, n_end: int = 21):
        if isinstance(n_start, bool) or not isinstance(n_start, int):
            raise TypeError("n_start must be an integer")
        if isinstance(n_end, bool) or not isinstance(n_end, int):
            raise TypeError("n_end must be an integer")
        if n_start > n_end:
            raise ValueError("n_start must not exceed n_end")

        self._n_start = n_start
        self._n_end = n_end
        self._function_samples = {}
        self._derivative_samples = {}
        self._diff_methods = {}

    def add_sample(
        self,
        name: str,
        function:   ScalarFunction,
        derivative: ScalarFunction
    ) -> None:
        if name in self._function_samples:
            logger.warning("Overriding function [%s]", name)

        self._function_samples[name] = function
        self._derivative_samples[name] = derivative
                
    def add_method(
        self,
        name:   str,
        method: DifferentiationMethod
    ) -> None:
        if name in self._diff_methods:
            logger.warning("Overriding method [%s]", name)

        self._diff_methods[name] = method

    def run_all(
        self,
        x: float
    ) -> list[DifferentiationResult]:
        results: list[DifferentiationResult] = []

        for func_name, func in self._function_samples.items():
            derivative_exact = self._derivative_samples[func_name](x)

            for method_name, method in self._diff_methods.items():
                for n in range(self._n_start, self._n_end + 1):
                    h = 2.0 ** (1 - n)
                    if h == 0.0:
                        raise ValueError(f"step underflow for n={n}")

                    derivative_numeric = method(func, x, h)
                    results.append(
                        DifferentiationResult(
                            function_name=func_name,
                            method_name=method_name,
                            x=x,
                            n=n,
                            h=h,
                            derivative_exact=derivative_exact,
                            derivative_numeric=derivative_numeric,
                            abs_error=abs(derivative_numeric - derivative_exact),
                        )
                    )

        return results
