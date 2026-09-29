# Robust and Distributionally Robust Optimization

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary Jors Academy methodology umbrella for robust optimization (RO), distributionally robust optimization (DRO), and data-driven uncertainty sets. It provides transparent reference implementations and out-of-sample evaluation patterns that complement application-specific robust models elsewhere in the portfolio.

<!-- portfolio-umbrella:end -->

The central research question is:

> How should optimization decisions change when model parameters or probability distributions are uncertain, and how should that robustness be evaluated out of sample?

## Native research modules

| Module | Uncertainty model | Benchmark |
|---|---|---|
| Budgeted robust optimization | Bertsimas-Sim cardinality/budget set | Production planning under uncertain resource consumption |
| Wasserstein DRO | Finite-support 1-Wasserstein ambiguity set | Newsvendor under demand-distribution uncertainty |

The implementations deliberately separate three quantities that are often conflated:

1. nominal or empirical objective value;
2. worst-case objective value inside the declared uncertainty/ambiguity set;
3. out-of-sample realized performance on fresh data.

## Method map

```text
Optimization under uncertainty
├── Robust optimization
│   ├── Box / interval uncertainty
│   ├── Budgeted polyhedral uncertainty
│   ├── Ellipsoidal uncertainty
│   └── Adjustable robust optimization
├── Chance-constrained optimization
├── Distributionally robust optimization
│   ├── Wasserstein ambiguity sets
│   ├── Moment-based ambiguity sets
│   └── Divergence-based ambiguity sets
└── Data-driven uncertainty sets
    ├── Bootstrap / scenario sets
    └── Conformal uncertainty sets
```

## Installation

```bash
python -m pip install -e ".[dev]"
pytest
```

Python 3.10+ is supported. NumPy and SciPy/HiGHS are sufficient for all committed examples.

## Run

```bash
python -m robustopt.budgeted_production
python -m robustopt.wasserstein_newsvendor
```

## Research standard

Every benchmark should state:

- the nominal model;
- the uncertainty or ambiguity set;
- the robustness parameter and its units;
- the deterministic or tractable reformulation used;
- the price of robustness;
- an out-of-sample evaluation protocol;
- sensitivity to the robustness radius/budget.

A more conservative objective is not automatically a better decision. Robustness is evaluated by the trade-off among nominal performance, tail/worst-case performance, feasibility, and stability under distribution shift.

## Roadmap

Planned extensions include:

- ellipsoidal robust SOCP formulations;
- adjustable robust optimization;
- chance constraints and scenario approximation;
- moment-based DRO;
- phi-divergence DRO;
- conformal uncertainty sets;
- Wasserstein DRO dual reformulations;
- decision-focused calibration of ambiguity-set size;
- robust and DRO models solved with decomposition.

## License

Research and educational use. Add a repository-level license before redistributing or using the code under a specific licensing regime.
