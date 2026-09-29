import numpy as np

from robustopt.budgeted_production import (
    example_instance,
    solve_budgeted_robust,
)
from robustopt.wasserstein_newsvendor import (
    empirical_expected_cost,
    example_data,
    solve_wasserstein_newsvendor,
    worst_case_cost,
)


def test_budgeted_robust_solution_is_protected():
    instance = example_instance()
    for gamma in (0.0, 0.5, 1.0, 2.0, 3.0):
        result = solve_budgeted_robust(gamma, instance)
        assert result.worst_case_consumption <= instance.capacity + 1e-8


def test_profit_is_nonincreasing_with_robustness_budget():
    profits = [
        solve_budgeted_robust(gamma).profit
        for gamma in (0.0, 0.5, 1.0, 2.0, 3.0)
    ]
    assert all(a + 1e-8 >= b for a, b in zip(profits, profits[1:]))


def test_wasserstein_zero_radius_matches_empirical_cost():
    samples, support = example_data()
    for q in (7.0, 9.0, 12.0):
        worst, _ = worst_case_cost(q, samples, support, 0.0, 4.0, 1.0)
        empirical = empirical_expected_cost(q, samples, 4.0, 1.0)
        assert np.isclose(worst, empirical, atol=1e-8)


def test_worst_case_value_is_nondecreasing_in_radius():
    samples, support = example_data()
    q = 9.0
    values = [
        worst_case_cost(q, samples, support, eps, 4.0, 1.0)[0]
        for eps in (0.0, 0.5, 1.5, 3.0)
    ]
    assert all(a <= b + 1e-8 for a, b in zip(values, values[1:]))


def test_dro_solver_returns_probability_distribution():
    samples, support = example_data()
    result = solve_wasserstein_newsvendor(samples, support, 1.0)
    assert np.isclose(result.worst_case_distribution.sum(), 1.0)
    assert np.all(result.worst_case_distribution >= -1e-10)
