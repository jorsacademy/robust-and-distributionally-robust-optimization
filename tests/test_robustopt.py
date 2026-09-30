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


from robustopt.ellipsoidal_production import (
    nominal_reference as ellipsoidal_nominal_reference,
    solve_ellipsoidal_robust,
)
from robustopt.gaussian_chance_production import (
    nominal_reference as chance_nominal_reference,
    solve_gaussian_chance_constraint,
)


def test_ellipsoidal_radius_zero_matches_nominal_lp():
    nominal_profit, _ = ellipsoidal_nominal_reference()
    result = solve_ellipsoidal_robust(0.0)
    assert np.isclose(result.profit, nominal_profit, atol=1e-6)
    assert result.worst_case_consumption <= 18.0 + 1e-7


def test_ellipsoidal_profit_is_nonincreasing_in_radius():
    profits = [solve_ellipsoidal_robust(r).profit for r in (0.0, 0.5, 1.0, 1.5)]
    assert all(a + 1e-6 >= b for a, b in zip(profits, profits[1:]))


def test_gaussian_alpha_half_matches_nominal_lp():
    nominal_profit, _ = chance_nominal_reference()
    result = solve_gaussian_chance_constraint(0.5)
    assert np.isclose(result.quantile, 0.0, atol=1e-12)
    assert np.isclose(result.profit, nominal_profit, atol=1e-6)


def test_gaussian_chance_constraint_meets_declared_risk():
    for alpha in (0.2, 0.1, 0.05):
        result = solve_gaussian_chance_constraint(alpha)
        assert result.deterministic_lhs <= 18.0 + 1e-7
        assert result.implied_violation_probability <= alpha + 1e-7


def test_profit_falls_as_service_confidence_tightens():
    alphas = (0.5, 0.2, 0.1, 0.05)
    profits = [solve_gaussian_chance_constraint(alpha).profit for alpha in alphas]
    assert all(a + 1e-6 >= b for a, b in zip(profits, profits[1:]))


from robustopt.adaptive_wasserstein_newsvendor import (
    ambiguity_radius,
    example_data as adaptive_example_data,
    solve_adaptive_wasserstein_newsvendor,
)
from robustopt.fuzzy_possibilistic_production import (
    solve_fuzzy_production,
    triangular_effective_capacity,
)


def test_adaptive_dro_radius_shrinks_with_sample_size():
    radii = [ambiguity_radius(n, scale=2.0, floor=0.1) for n in range(4, 12)]
    assert all(a >= b - 1e-12 for a, b in zip(radii, radii[1:]))


def test_adaptive_dro_updates_sample_size_and_returns_finite_values():
    initial, arriving, support = adaptive_example_data()
    steps = solve_adaptive_wasserstein_newsvendor(
        initial,
        arriving,
        support,
        radius_scale=2.0,
        radius_floor=0.1,
    )
    assert len(steps) == len(arriving)
    assert [s.sample_size for s in steps] == list(
        range(len(initial) + 1, len(initial) + len(arriving) + 1)
    )
    assert all(np.isfinite(s.worst_case_cost) for s in steps)
    assert all(s.epsilon >= 0.1 for s in steps)


def test_fuzzy_necessity_is_more_conservative_than_possibility():
    for level in (0.25, 0.5, 0.75, 1.0):
        nec = triangular_effective_capacity(12.0, 18.0, 24.0, level, "necessity")
        pos = triangular_effective_capacity(12.0, 18.0, 24.0, level, "possibility")
        assert nec <= pos + 1e-12


def test_fuzzy_production_respects_effective_capacity():
    for semantics in ("necessity", "possibility"):
        for level in (0.25, 0.5, 0.75, 1.0):
            result = solve_fuzzy_production(level, semantics)
            assert result.resource_consumption <= result.effective_capacity + 1e-8


def test_fuzzy_profit_is_nonincreasing_as_confidence_tightens():
    for semantics in ("necessity", "possibility"):
        profits = [
            solve_fuzzy_production(level, semantics).profit
            for level in (0.25, 0.5, 0.75, 1.0)
        ]
        assert all(a + 1e-8 >= b for a, b in zip(profits, profits[1:]))
