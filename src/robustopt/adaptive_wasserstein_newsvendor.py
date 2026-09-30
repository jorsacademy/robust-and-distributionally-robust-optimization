"""Data-adaptive Wasserstein DRO for a sequential newsvendor.

This module implements a compact adaptive-DRO benchmark: the empirical demand
distribution is updated as observations arrive, and the Wasserstein ambiguity
radius shrinks with sample size. At each decision epoch, the current ambiguity
set is solved with the finite-support Wasserstein newsvendor implementation in
`wasserstein_newsvendor`.

This is intentionally a data-adaptive / rolling re-estimation model. It is not
claimed to be a time-consistent nested multistage DRO formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wasserstein_newsvendor import solve_wasserstein_newsvendor


@dataclass
class AdaptiveDROStep:
    period: int
    sample_size: int
    epsilon: float
    order_quantity: float
    worst_case_cost: float
    empirical_cost: float


def ambiguity_radius(
    sample_size: int,
    scale: float = 2.0,
    floor: float = 0.1,
) -> float:
    """Return epsilon_n = max(floor, scale / sqrt(n))."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if scale < 0:
        raise ValueError("scale must be nonnegative")
    if floor < 0:
        raise ValueError("floor must be nonnegative")
    return float(max(floor, scale / np.sqrt(sample_size)))


def solve_adaptive_wasserstein_newsvendor(
    initial_samples: np.ndarray,
    arriving_demands: np.ndarray,
    support: np.ndarray,
    radius_scale: float = 2.0,
    radius_floor: float = 0.1,
    underage_cost: float = 4.0,
    overage_cost: float = 1.0,
    candidate_orders: np.ndarray | None = None,
) -> list[AdaptiveDROStep]:
    """Re-estimate and resolve a Wasserstein DRO model as data arrive.

    Each arriving demand is observed before that period's optimization solve.
    The returned sequence therefore documents how the ambiguity set and optimal
    order react to an expanding empirical sample.
    """
    samples = np.asarray(initial_samples, dtype=float).reshape(-1).tolist()
    arrivals = np.asarray(arriving_demands, dtype=float).reshape(-1)
    support = np.asarray(support, dtype=float).reshape(-1)

    if len(samples) == 0:
        raise ValueError("initial_samples must contain at least one observation")
    if support.size == 0:
        raise ValueError("support must contain at least one value")

    steps: list[AdaptiveDROStep] = []
    for period, demand in enumerate(arrivals, start=1):
        samples.append(float(demand))
        epsilon = ambiguity_radius(
            len(samples),
            scale=radius_scale,
            floor=radius_floor,
        )
        result = solve_wasserstein_newsvendor(
            samples=np.asarray(samples, dtype=float),
            support=support,
            epsilon=epsilon,
            underage_cost=underage_cost,
            overage_cost=overage_cost,
            candidate_orders=candidate_orders,
        )
        steps.append(
            AdaptiveDROStep(
                period=period,
                sample_size=len(samples),
                epsilon=epsilon,
                order_quantity=result.order_quantity,
                worst_case_cost=result.worst_case_cost,
                empirical_cost=result.empirical_cost,
            )
        )
    return steps


def example_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial = np.array([7.0, 8.0, 9.0, 10.0])
    arriving = np.array([12.0, 11.0, 8.0, 7.0, 13.0])
    support = np.arange(4.0, 16.0)
    return initial, arriving, support


def main() -> None:
    initial, arriving, support = example_data()
    for step in solve_adaptive_wasserstein_newsvendor(
        initial,
        arriving,
        support,
    ):
        print(
            {
                "period": step.period,
                "sample_size": step.sample_size,
                "epsilon": round(step.epsilon, 4),
                "q": step.order_quantity,
                "worst_case_cost": round(step.worst_case_cost, 4),
                "empirical_cost": round(step.empirical_cost, 4),
            }
        )


if __name__ == "__main__":
    main()
