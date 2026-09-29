"""Finite-support Wasserstein DRO for a single-period newsvendor.

For a fixed order quantity q, the inner adversary transports empirical mass from
observed samples d_i to a declared finite support z_j. The transport budget is

    sum_ij |d_i-z_j| pi_ij <= epsilon,

with each empirical observation carrying mass 1/n. This is an explicit
finite-support 1-Wasserstein ambiguity set. The outer problem evaluates a
declared candidate grid of order quantities and selects the q with the smallest
worst-case expected cost.

The finite support is a modeling assumption. It should be expanded or replaced
by an analytical dual reformulation when an unbounded/continuous support is
intended.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class WassersteinResult:
    epsilon: float
    order_quantity: float
    worst_case_cost: float
    empirical_cost: float
    worst_case_distribution: np.ndarray


def newsvendor_cost(
    order_quantity: float,
    demand: np.ndarray,
    underage_cost: float,
    overage_cost: float,
) -> np.ndarray:
    demand = np.asarray(demand, dtype=float)
    shortage = np.maximum(demand - order_quantity, 0.0)
    excess = np.maximum(order_quantity - demand, 0.0)
    return underage_cost * shortage + overage_cost * excess


def empirical_expected_cost(
    order_quantity: float,
    samples: np.ndarray,
    underage_cost: float,
    overage_cost: float,
) -> float:
    return float(
        newsvendor_cost(order_quantity, samples, underage_cost, overage_cost).mean()
    )


def worst_case_cost(
    order_quantity: float,
    samples: np.ndarray,
    support: np.ndarray,
    epsilon: float,
    underage_cost: float,
    overage_cost: float,
) -> tuple[float, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    support = np.asarray(support, dtype=float)
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")

    n = samples.size
    m = support.size
    costs = newsvendor_cost(
        order_quantity, support, underage_cost, overage_cost
    )

    # pi[i,j] transports probability mass from empirical sample i to support j.
    # Maximize sum pi*c  -> minimize negative cost.
    c = -np.tile(costs, n)

    a_eq = np.zeros((n, n * m))
    for i in range(n):
        a_eq[i, i * m : (i + 1) * m] = 1.0
    b_eq = np.full(n, 1.0 / n)

    distances = np.abs(samples[:, None] - support[None, :]).reshape(-1)
    a_ub = distances.reshape(1, -1)
    b_ub = np.array([epsilon])

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Wasserstein inner problem failed: {result.message}")

    transport = np.asarray(result.x).reshape(n, m)
    distribution = transport.sum(axis=0)
    return float(-result.fun), distribution


def solve_wasserstein_newsvendor(
    samples: np.ndarray,
    support: np.ndarray,
    epsilon: float,
    underage_cost: float = 4.0,
    overage_cost: float = 1.0,
    candidate_orders: np.ndarray | None = None,
) -> WassersteinResult:
    samples = np.asarray(samples, dtype=float)
    support = np.asarray(support, dtype=float)
    candidate_orders = (
        support.copy()
        if candidate_orders is None
        else np.asarray(candidate_orders, dtype=float)
    )

    best_q = None
    best_wc = np.inf
    best_dist = None

    for q in candidate_orders:
        wc, dist = worst_case_cost(
            float(q),
            samples,
            support,
            epsilon,
            underage_cost,
            overage_cost,
        )
        if wc < best_wc - 1e-12:
            best_q = float(q)
            best_wc = float(wc)
            best_dist = dist

    if best_q is None or best_dist is None:
        raise RuntimeError("No candidate order quantity was evaluated.")

    return WassersteinResult(
        epsilon=float(epsilon),
        order_quantity=best_q,
        worst_case_cost=best_wc,
        empirical_cost=empirical_expected_cost(
            best_q, samples, underage_cost, overage_cost
        ),
        worst_case_distribution=np.asarray(best_dist),
    )


def solve_saa_newsvendor(
    samples: np.ndarray,
    candidate_orders: np.ndarray,
    underage_cost: float = 4.0,
    overage_cost: float = 1.0,
) -> tuple[float, float]:
    values = [
        empirical_expected_cost(float(q), samples, underage_cost, overage_cost)
        for q in candidate_orders
    ]
    index = int(np.argmin(values))
    return float(candidate_orders[index]), float(values[index])


def evaluate_out_of_sample(
    order_quantity: float,
    demand: np.ndarray,
    underage_cost: float = 4.0,
    overage_cost: float = 1.0,
) -> dict[str, float]:
    costs = newsvendor_cost(
        order_quantity, np.asarray(demand), underage_cost, overage_cost
    )
    return {
        "mean_cost": float(np.mean(costs)),
        "p90_cost": float(np.quantile(costs, 0.90)),
        "max_cost": float(np.max(costs)),
    }


def example_data() -> tuple[np.ndarray, np.ndarray]:
    samples = np.array([7.0, 8.0, 8.0, 9.0, 10.0, 12.0])
    support = np.arange(4.0, 16.0)
    return samples, support


def main() -> None:
    samples, support = example_data()
    saa_q, saa_cost = solve_saa_newsvendor(samples, support)
    print({"SAA_q": saa_q, "SAA_empirical_cost": saa_cost})

    for epsilon in (0.0, 0.5, 1.5, 3.0):
        result = solve_wasserstein_newsvendor(samples, support, epsilon)
        print(
            {
                "epsilon": epsilon,
                "q": result.order_quantity,
                "worst_case_cost": result.worst_case_cost,
                "empirical_cost": result.empirical_cost,
            }
        )


if __name__ == "__main__":
    main()
