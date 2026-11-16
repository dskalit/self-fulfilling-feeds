"""Outcome measures. Each function takes one snapshot of opinions (an array
of length N); summarize() applies them to a whole run over time.

What we look at and why
-----------------------
group means / gap   Do men and women end up thinking differently, or
                    does one group get pulled toward the other?
drift               How far each group moved from its own original
                    opinions theta. Women drifting toward the men's side
                    is the effect we're looking for.
spread              Std of opinions within each group. Shrinkage toward
                    the pooled mean should show up as compression.
polarization        Three simple numbers, read together: variance of all
                    opinions, extremity (mean |x|), and the share of agents
                    on the positive side. Polarized = both sides populated
                    (share near 0.5) AND opinions far from 0. Radicalized =
                    share near 0 or 1. (We tried Sarle's bimodality
                    coefficient, but Baumann's camps have long tails of very
                    extreme, very active agents, which inflates kurtosis and
                    makes the coefficient LOW for clearly polarized runs.)
prediction error    How wrong the platform's x_hat is, measured two ways:
                      vs current opinions x   (what the platform can check:
                                               does the user engage as predicted?)
                      vs original opinions theta  (what users "really" thought)
                    The self-fulfilling signature is the first error FALLING
                    while drift RISES: the platform looks more and more
                    accurate because it moved people toward its predictions.
"""

import numpy as np

from sfp.population import MEN, WOMEN


def group_means(x, group):
    """Mean opinion of men and of women, as (men, women)."""
    return x[group == MEN].mean(), x[group == WOMEN].mean()


def group_gap(x, group):
    """Men's mean minus women's mean. Shrinks toward 0 if the groups converge."""
    m, w = group_means(x, group)
    return m - w


def drift(x, theta, group):
    """Average change from original opinion, per group, as (men, women).
    Positive = moved toward the positive side of the opinion axis."""
    d = x - theta
    return d[group == MEN].mean(), d[group == WOMEN].mean()


def group_spread(x, group):
    """Standard deviation of opinions within each group, as (men, women)."""
    return x[group == MEN].std(), x[group == WOMEN].std()


def extremity(x):
    """Mean distance from the neutral opinion 0."""
    return np.abs(x).mean()


def share_positive(x):
    """Fraction of agents with a positive opinion. ~0.5 with two camps,
    ~0 or ~1 when everyone has been pulled to one side."""
    return np.mean(x > 0)


def prediction_error(x_hat, target, mask=None):
    """Mean absolute difference between prediction and target,
    optionally only over agents where mask is True (e.g. only women)."""
    err = np.abs(x_hat - target)
    return err.mean() if mask is None else err[mask].mean()


def summarize(result):
    """Apply all metrics to every snapshot of a run from runner.run().

    Returns a dict of arrays over time (same length as result['times']).
    Prediction errors are NaN when the platform makes no predictions.
    """
    x, theta, group = result["x"], result["theta"], result["group"]
    x_hat = result["x_hat"]
    women = group == WOMEN
    R = len(result["times"])

    out = {k: np.full(R, np.nan) for k in [
        "mean_men", "mean_women", "gap", "drift_men", "drift_women",
        "spread_men", "spread_women", "variance", "extremity", "share_positive",
        "err_vs_x", "err_vs_theta", "err_vs_x_women", "err_vs_theta_women"]}

    for t in range(R):
        xt = x[t]
        out["mean_men"][t], out["mean_women"][t] = group_means(xt, group)
        out["gap"][t] = group_gap(xt, group)
        out["drift_men"][t], out["drift_women"][t] = drift(xt, theta, group)
        out["spread_men"][t], out["spread_women"][t] = group_spread(xt, group)
        out["variance"][t] = xt.var()
        out["extremity"][t] = extremity(xt)
        out["share_positive"][t] = share_positive(xt)
        if x_hat is not None:
            out["err_vs_x"][t] = prediction_error(x_hat[t], xt)
            out["err_vs_theta"][t] = prediction_error(x_hat[t], theta)
            out["err_vs_x_women"][t] = prediction_error(x_hat[t], xt, women)
            out["err_vs_theta_women"][t] = prediction_error(x_hat[t], theta, women)
    return out
