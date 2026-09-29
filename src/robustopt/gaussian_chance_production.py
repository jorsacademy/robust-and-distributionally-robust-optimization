"""Gaussian chance-constrained production.

Let uncertain resource coefficients A be jointly normal with independent
component standard deviations sigma_i. For nonnegative production x,

    A'x ~ Normal(mu'x, ||sigma*x||_2^2).

Therefore

    P(A'x <= capacity) >= 1 - alpha

is exactly equivalent (for alpha <= 0.5) to

    mu'x + Phi^{-1}(1-alpha) ||sigma*x||_2 <= capacity.

The deterministic equivalent is solved with SLSQP and the implied analytical
violation probability is reported independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.stats import norm


@dataclass(frozen=True)
class ChanceInstance:
    profit: np.ndarray
    mean_use: np.ndarray
    std_use: np.ndarray
    capacity: float
    upper_bounds: np.ndarray


@dataclass
class ChanceResult:
    alpha: float
    confidence: float
    quantile: float
    production: np.ndarray
    profit: float
    mean_consumption: float
    consumption_std: float
    deterministic_lhs: float
    implied_violation_probability: float


def example_instance() -> ChanceInstance:
    return ChanceInstance(
        profit=np.array([8.0, 7.0, 6.0]),
        mean_use=np.array([3.0, 2.0, 2.0]),
        std_use=np.array([1.0, 1.5, 0.5]),
        capacity=18.0,
        upper_bounds=np.array([5.0, 5.0, 5.0]),
    )


def nominal_reference(instance: ChanceInstance | None = None) -> tuple[float, np.ndarray]:
    instance = instance or example_instance()
    result = linprog(
        -instance.profit,
        A_ub=instance.mean_use.reshape(1, -1),
        b_ub=np.array([instance.capacity]),
        bounds=[(0.0, float(u)) for u in instance.upper_bounds],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Nominal LP failed: {result.message}")
    return float(-result.fun), np.asarray(result.x)


def distribution_parameters(
    production: np.ndarray, instance: ChanceInstance
) -> tuple[float, float]:
    x = np.asarray(production, dtype=float)
    mean = float(instance.mean_use @ x)
    std = float(np.linalg.norm(instance.std_use * x, 2))
    return mean, std


def solve_gaussian_chance_constraint(
    alpha: float,
    instance: ChanceInstance | None = None,
) -> ChanceResult:
    instance = instance or example_instance()
    if not 0.0 < alpha <= 0.5:
        raise ValueError("alpha must lie in (0, 0.5] for this convex reference model")

    z = float(norm.ppf(1.0 - alpha))
    _, x0 = nominal_reference(instance)

    def lhs(x: np.ndarray) -> float:
        mean, std = distribution_parameters(x, instance)
        return mean + z * std

    lhs0 = lhs(x0)
    if lhs0 > instance.capacity and lhs0 > 0:
        x0 = x0 * (instance.capacity / lhs0) * 0.98

    result = minimize(
        lambda x: float(-instance.profit @ x),
        x0,
        method="SLSQP",
        bounds=[(0.0, float(u)) for u in instance.upper_bounds],
        constraints=[{"type": "ineq", "fun": lambda x: instance.capacity - lhs(x)}],
        options={"ftol": 1e-10, "maxiter": 2000, "disp": False},
    )
    x = np.asarray(result.x)
    checked_lhs = lhs(x)
    if not result.success and checked_lhs > instance.capacity + 1e-7:
        raise RuntimeError(f"Chance-constrained solve failed: {result.message}")
    mean, std = distribution_parameters(x, instance)
    deterministic_lhs = mean + z * std
    if std <= 1e-14:
        violation = 0.0 if mean <= instance.capacity else 1.0
    else:
        violation = float(1.0 - norm.cdf((instance.capacity - mean) / std))

    return ChanceResult(
        alpha=float(alpha),
        confidence=float(1.0 - alpha),
        quantile=z,
        production=x,
        profit=float(instance.profit @ x),
        mean_consumption=mean,
        consumption_std=std,
        deterministic_lhs=deterministic_lhs,
        implied_violation_probability=violation,
    )


def main() -> None:
    for alpha in (0.5, 0.2, 0.1, 0.05):
        result = solve_gaussian_chance_constraint(alpha)
        print(
            {
                "alpha": alpha,
                "confidence": result.confidence,
                "profit": result.profit,
                "production": result.production.round(5).tolist(),
                "implied_violation_probability": result.implied_violation_probability,
                "deterministic_lhs": result.deterministic_lhs,
            }
        )


if __name__ == "__main__":
    main()
