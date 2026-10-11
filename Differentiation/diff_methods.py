from collections.abc import Callable

from .diff_engine import DifferentiationMethod

""" All of functions below are following DifferentiationMethod signature,
    and implement a step in calculation of f at point x, following various methods.

    Args:
        f: Function to differentiate.
        x: Point at which to approximate the derivative.
        h: Nonzero differentiation step.

    Returns:
        Approximation of f'(x).
"""

def forward_difference_calc(
    f: Callable[[float], float],
    x: float,
    h: float
):
    return (f(x + h) - f(x)) / h


def backward_difference_calc(
    f: Callable[[float], float],
    x: float,
    h: float
):
    return (f(x) - f(x - h)) / h


def central_difference_calc(
    f: Callable[[float], float],
    x: float,
    h: float
):
    return (f(x + h) - f(x - h)) / (2 * h)


def fourth_order_central_calc(
    f: Callable[[float], float],
    x: float,
    h: float
):
    central_h = (f(x + h) - f(x - h)) / (2 * h)
    central_2h = (f(x + 2 * h) - f(x - 2 * h)) / (4 * h)
    return (4 / 3) * central_h - (1 / 3) * central_2h


def sixth_order_central_calc(
    f: Callable[[float], float],
    x: float,
    h: float
):
    central_h = (f(x + h) - f(x - h)) / (2 * h)
    central_2h = (f(x + 2 * h) - f(x - 2 * h)) / (4 * h)
    central_3h = (f(x + 3 * h) - f(x - 3 * h)) / (6 * h)
    return (
        (3 / 2) * central_h
        - (3 / 5) * central_2h
        + (1 / 10) * central_3h
    )


differentiation_methods: dict[str, DifferentiationMethod] = {
    "D_1": forward_difference_calc,
    "D_2": backward_difference_calc,
    "D_3": central_difference_calc,
    "D_4": fourth_order_central_calc,
    "D_5": sixth_order_central_calc,
}
