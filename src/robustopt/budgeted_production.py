"""Bertsimas-Sim budgeted robust counterpart for a production LP.

Nominal model
-------------
maximize p'x
subject to a'x <= capacity
           0 <= x <= upper

The resource coefficients are uncertain:

    a_i -> a_i + d_i z_i,
    0 <= z_i <= 1,
    sum_i z_i <= Gamma.

The support function of this budgeted uncertainty set yields the linear robust
counterpart

    a'x + Gamma*rho + sum_i q_i <= capacity
    rho + q_i >= d_i x_i
    rho, q >= 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class ProductionInstance:
    profit: np.ndarray
    nominal_use: np.ndarray
    deviation: np.ndarray
    capacity: float
    upper_bounds: np.ndarray


@dataclass
class RobustProductionResult:
    gamma: float
    production: np.ndarray
    profit: float
    nominal_consumption: float
    worst_case_consumption: float
    protection_term: float


def example_instance() -> ProductionInstance:
    return ProductionInstance(
        profit=np.array([8.0, 7.0, 6.0]),
        nominal_use=np.array([3.0, 2.0, 2.0]),
        deviation=np.array([1.0, 1.5, 0.5]),
        capacity=18.0,
        upper_bounds=np.array([5.0, 5.0, 5.0]),
    )


def budgeted_support(values: np.ndarray, gamma: float) -> float:
    """Maximize values'z over 0<=z<=1 and sum(z)<=gamma."""

    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    if gamma <= 0:
        return 0.0

    whole = int(np.floor(gamma))
    fraction = float(gamma - whole)
    ordered = np.sort(values)[::-1]

    support = float(ordered[:whole].sum()) if whole else 0.0
    if whole < ordered.size:
        support += fraction * float(ordered[whole])
    return support


def solve_budgeted_robust(
    gamma: float,
    instance: ProductionInstance | None = None,
) -> RobustProductionResult:
    instance = instance or example_instance()
    n = instance.profit.size
    if not 0.0 <= gamma <= float(n):
        raise ValueError(f"gamma must lie in [0, {n}]")

    # Variables: x[0:n], rho, q[0:n]
    n_vars = 2 * n + 1
    rho_index = n
    q_start = n + 1

    c = np.zeros(n_vars)
    c[:n] = -instance.profit

    a_ub = []
    b_ub = []

    # nominal_use'x + Gamma*rho + sum(q) <= capacity
    row = np.zeros(n_vars)
    row[:n] = instance.nominal_use
    row[rho_index] = gamma
    row[q_start:] = 1.0
    a_ub.append(row)
    b_ub.append(instance.capacity)

    # deviation_i*x_i - rho - q_i <= 0
    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = instance.deviation[i]
        row[rho_index] = -1.0
        row[q_start + i] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

    bounds = [(0.0, float(instance.upper_bounds[i])) for i in range(n)]
    bounds += [(0.0, None)]  # rho
    bounds += [(0.0, None)] * n

    result = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Robust production LP failed: {result.message}")

    x = np.asarray(result.x[:n])
    nominal = float(instance.nominal_use @ x)
    protection = budgeted_support(instance.deviation * x, gamma)
    worst = nominal + protection

    return RobustProductionResult(
        gamma=float(gamma),
        production=x,
        profit=float(instance.profit @ x),
        nominal_consumption=nominal,
        worst_case_consumption=worst,
        protection_term=protection,
    )


def price_of_robustness(
    gamma: float, instance: ProductionInstance | None = None
) -> float:
    instance = instance or example_instance()
    nominal = solve_budgeted_robust(0.0, instance)
    robust = solve_budgeted_robust(gamma, instance)
    return nominal.profit - robust.profit


def main() -> None:
    for gamma in (0.0, 1.0, 2.0, 3.0):
        result = solve_budgeted_robust(gamma)
        print(
            {
                "gamma": gamma,
                "profit": result.profit,
                "production": result.production.round(4).tolist(),
                "nominal_consumption": result.nominal_consumption,
                "worst_case_consumption": result.worst_case_consumption,
                "price_of_robustness": price_of_robustness(gamma),
            }
        )


if __name__ == "__main__":
    main()
