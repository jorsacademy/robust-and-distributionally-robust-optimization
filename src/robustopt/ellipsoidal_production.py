"""Ellipsoidal robust production with a second-order-cone-type reformulation.

Resource coefficients follow

    a(u) = a_bar + diag(d) u,  ||u||_2 <= radius.

For a fixed nonnegative production vector x, worst-case resource use is

    a_bar'x + radius * ||diag(d) x||_2.

The resulting convex robust problem is solved with SciPy SLSQP. The module
reports the analytical worst-case left-hand side independently of the solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass(frozen=True)
class EllipsoidalInstance:
    profit: np.ndarray
    nominal_use: np.ndarray
    deviation_scale: np.ndarray
    capacity: float
    upper_bounds: np.ndarray


@dataclass
class EllipsoidalResult:
    radius: float
    production: np.ndarray
    profit: float
    nominal_consumption: float
    protection_term: float
    worst_case_consumption: float
    success: bool


def example_instance() -> EllipsoidalInstance:
    return EllipsoidalInstance(
        profit=np.array([8.0, 7.0, 6.0]),
        nominal_use=np.array([3.0, 2.0, 2.0]),
        deviation_scale=np.array([1.0, 1.5, 0.5]),
        capacity=18.0,
        upper_bounds=np.array([5.0, 5.0, 5.0]),
    )


def robust_consumption(
    production: np.ndarray, radius: float, instance: EllipsoidalInstance
) -> tuple[float, float, float]:
    x = np.asarray(production, dtype=float)
    nominal = float(instance.nominal_use @ x)
    protection = float(radius * np.linalg.norm(instance.deviation_scale * x, 2))
    return nominal, protection, nominal + protection


def nominal_reference(
    instance: EllipsoidalInstance | None = None,
) -> tuple[float, np.ndarray]:
    instance = instance or example_instance()
    result = linprog(
        -instance.profit,
        A_ub=instance.nominal_use.reshape(1, -1),
        b_ub=np.array([instance.capacity]),
        bounds=[(0.0, float(u)) for u in instance.upper_bounds],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Nominal LP failed: {result.message}")
    return float(-result.fun), np.asarray(result.x)


def solve_ellipsoidal_robust(
    radius: float,
    instance: EllipsoidalInstance | None = None,
) -> EllipsoidalResult:
    instance = instance or example_instance()
    if radius < 0:
        raise ValueError("radius must be nonnegative")

    _, x0 = nominal_reference(instance)
    _, _, lhs0 = robust_consumption(x0, radius, instance)
    if lhs0 > instance.capacity and lhs0 > 0:
        x0 = x0 * (instance.capacity / lhs0) * 0.98

    def objective(x: np.ndarray) -> float:
        return float(-instance.profit @ x)

    def feasibility(x: np.ndarray) -> float:
        return instance.capacity - robust_consumption(x, radius, instance)[2]

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, float(u)) for u in instance.upper_bounds],
        constraints=[{"type": "ineq", "fun": feasibility}],
        options={"ftol": 1e-10, "maxiter": 2000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Ellipsoidal robust solve failed: {result.message}")

    x = np.asarray(result.x)
    nominal, protection, worst = robust_consumption(x, radius, instance)
    return EllipsoidalResult(
        radius=float(radius),
        production=x,
        profit=float(instance.profit @ x),
        nominal_consumption=nominal,
        protection_term=protection,
        worst_case_consumption=worst,
        success=bool(result.success),
    )


def main() -> None:
    nominal_profit, _ = nominal_reference()
    for radius in (0.0, 0.5, 1.0, 2.0):
        result = solve_ellipsoidal_robust(radius)
        print(
            {
                "radius": radius,
                "profit": result.profit,
                "price_of_robustness": nominal_profit - result.profit,
                "production": result.production.round(5).tolist(),
                "worst_case_consumption": result.worst_case_consumption,
            }
        )


if __name__ == "__main__":
    main()
