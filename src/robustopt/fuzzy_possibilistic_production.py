"""Fuzzy and possibilistic production planning with triangular capacity.

A triangular fuzzy capacity C=(l,m,u) is represented by its membership
function. For a one-sided resource constraint, the module uses standard
possibility / necessity cuts:

    N(C >= z) >= eta  ->  z <= m - eta (m-l)
    Pi(C >= z) >= eta ->  z <= u - eta (u-m)

for eta in (0, 1]. The resulting deterministic LP is solved with HiGHS through
SciPy. The example is deliberately small so the difference between necessity
and possibility semantics is transparent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class FuzzyProductionResult:
    semantics: str
    level: float
    effective_capacity: float
    production: np.ndarray
    profit: float
    resource_consumption: float


def triangular_effective_capacity(
    lower: float,
    modal: float,
    upper: float,
    level: float,
    semantics: str = "necessity",
) -> float:
    """Convert a triangular fuzzy capacity into a deterministic cut."""
    if not lower <= modal <= upper:
        raise ValueError("Require lower <= modal <= upper")
    if not 0.0 < level <= 1.0:
        raise ValueError("level must lie in (0, 1]")

    semantics = semantics.lower()
    if semantics == "necessity":
        return float(modal - level * (modal - lower))
    if semantics == "possibility":
        return float(upper - level * (upper - modal))
    raise ValueError("semantics must be 'necessity' or 'possibility'")


def solve_fuzzy_production(
    level: float,
    semantics: str = "necessity",
    lower_capacity: float = 12.0,
    modal_capacity: float = 18.0,
    upper_capacity: float = 24.0,
) -> FuzzyProductionResult:
    """Solve a two-product fuzzy-capacity production LP."""
    profits = np.array([9.0, 7.0])
    resource_use = np.array([3.0, 2.0])
    bounds = [(0.0, 4.0), (0.0, 6.0)]

    capacity = triangular_effective_capacity(
        lower_capacity,
        modal_capacity,
        upper_capacity,
        level,
        semantics,
    )

    result = linprog(
        -profits,
        A_ub=resource_use.reshape(1, -1),
        b_ub=np.array([capacity]),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Fuzzy production LP failed: {result.message}")

    production = np.asarray(result.x, dtype=float)
    consumption = float(resource_use @ production)
    return FuzzyProductionResult(
        semantics=semantics.lower(),
        level=float(level),
        effective_capacity=capacity,
        production=production,
        profit=float(-result.fun),
        resource_consumption=consumption,
    )


def main() -> None:
    for semantics in ("necessity", "possibility"):
        for level in (0.25, 0.50, 0.75, 1.00):
            result = solve_fuzzy_production(level, semantics)
            print(
                {
                    "semantics": result.semantics,
                    "level": result.level,
                    "effective_capacity": result.effective_capacity,
                    "production": result.production.round(4).tolist(),
                    "profit": round(result.profit, 4),
                }
            )


if __name__ == "__main__":
    main()
