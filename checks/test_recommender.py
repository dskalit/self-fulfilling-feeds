"""Small tests of PredictiveRecommender on hand-made arrays.

Each test builds a tiny population (5 agents) where we know the right
answer by hand, and checks the recommender gets exactly that.

Run from the repo root:  python -m checks.test_recommender
"""

import numpy as np

from sfp.population import MEN, WOMEN
from sfp.recommender import (NoRecommender, PredictiveRecommender,
                             TrueOpinionMatcher, make_platform)

GROUP = np.array([MEN, MEN, WOMEN, WOMEN, MEN])
X = np.array([-0.5, 0.2, 0.7, -0.9, 0.4])


def mask(*idx):
    """Boolean 'active' array with True at the given positions."""
    m = np.zeros(len(X), bool)
    m[list(idx)] = True
    return m


def test_no_noise_no_shrinkage_recovers_x():
    """noise_sd=0 and lam=0: estimates() equals x for agents that were active."""
    rng = np.random.default_rng(0)
    p = PredictiveRecommender(GROUP, lam=0, noise_sd=0)
    active = mask(0, 1, 2)
    p.observe(X, active, rng)
    assert np.allclose(p.estimates()[active], X[active])


def test_unseen_agent_gets_pooled_mean():
    """An agent who has never been seen gets exactly the pooled mean."""
    rng = np.random.default_rng(0)
    p = PredictiveRecommender(GROUP, lam=5, noise_sd=0)
    p.observe(X, mask(0, 1, 2), rng)
    pooled = (X[0] + X[1] + X[2]) / 3
    assert np.isclose(p.pooled_mean(), pooled)
    assert np.allclose(p.estimates()[[3, 4]], pooled)


def test_pooled_mean_weighted_by_observations():
    """Agent 0 observed 3 times, agent 1 once: agent 0 counts 3x."""
    rng = np.random.default_rng(0)
    p = PredictiveRecommender(GROUP, lam=5, noise_sd=0)
    for _ in range(3):
        p.observe(X, mask(0), rng)
    p.observe(X, mask(1), rng)
    assert np.isclose(p.pooled_mean(), (3 * X[0] + X[1]) / 4)


def test_kappa_one_puts_target_group_on_x_target():
    """kappa=1 puts the target group exactly on x_target; others untouched."""
    rng = np.random.default_rng(0)
    p = PredictiveRecommender(GROUP, lam=5, noise_sd=0,
                              steer_group="women", kappa=1.0, x_target=0.3)
    p.observe(X, np.ones(len(X), bool), rng)
    match = p.matching_opinions(X)
    women = GROUP == WOMEN
    assert np.allclose(match[women], 0.3)
    assert np.allclose(match[~women], p.estimates()[~women])  # men not steered


def test_window_forgets_old_observations():
    """window=2: each step old observations keep weight 1 - 1/2 = 0.5.

    Agent 0 is seen at opinion a, then (one step later) at opinion b.
    Without memory loss its average is (a + b) / 2. With window=2 the old
    observation counts half, so the average is (0.5*a + b) / 1.5,
    i.e. closer to the recent opinion b.
    """
    rng = np.random.default_rng(0)
    a, b = -0.6, 0.6
    x1, x2 = X.copy(), X.copy()
    x1[0], x2[0] = a, b

    forget = PredictiveRecommender(GROUP, lam=0, noise_sd=0, window=2)
    keep = PredictiveRecommender(GROUP, lam=0, noise_sd=0, window=None)
    for p in (forget, keep):
        p.observe(x1, mask(0), rng)
        p.observe(x2, mask(0), rng)

    assert np.isclose(keep.estimates()[0], (a + b) / 2)
    assert np.isclose(forget.estimates()[0], (0.5 * a + b) / 1.5)
    assert np.isclose(forget.n[0], 1.5)   # observation count stays bounded


def test_make_platform_builds_each_kind():
    params = {"lam": 5.0, "noise_sd": 0.1, "window": None,
              "steer_group": None, "kappa": 0.0, "x_target": 0.0}
    assert isinstance(make_platform("none", GROUP, params), NoRecommender)
    assert isinstance(make_platform("true", GROUP, params), TrueOpinionMatcher)
    assert isinstance(make_platform("predictive", GROUP, params), PredictiveRecommender)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError:
            failed += 1
            print(f"FAIL  {t.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")


if __name__ == "__main__":
    main()
