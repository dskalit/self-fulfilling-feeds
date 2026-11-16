"""The platform: what it knows about users and how it matches them.

Three platforms, one per condition in our experiments. They all have the
same three methods, so dynamics.simulate() can use any of them.

  NoRecommender        partners picked at random (no matching at all)
  TrueOpinionMatcher   matches on true opinions; a perfect recommender.
                       This is plain Baumann and our control condition.
  PredictiveRecommender matches on estimated opinions x_hat, built from
                       noisy observations. This is where the bias comes from.

How PredictiveRecommender estimates opinions
--------------------------------------------
Each time an agent is active, the platform sees their opinion plus noise
(think: what they clicked, liked or commented). For agent i:

    own_mean_i = average of i's observations
    pooled     = average of ALL observations from everyone
    w_i        = n_i / (n_i + lam)       n_i = number of observations of i
    x_hat_i    = w_i * own_mean_i + (1 - w_i) * pooled

So an agent the platform has seen a lot is predicted mostly from their
own history; an agent it has barely seen is predicted mostly from the
pooled mean. This is standard shrinkage, what a sensible recommender
does with sparse data.

The gender mechanism is in the pooled mean: it averages over
OBSERVATIONS, not over users. Men produce ~75% of observations, so the
pooled mean sits close to the men's average. Women are observed less,
so they get shrunk more, and they get shrunk toward the men. Nobody
told the platform anything about gender; it falls out of who is active.

Steering
--------
On top of that, the platform can deliberately nudge the opinions it
matches on for a target group:

    x_hat_i <- x_hat_i + kappa * (x_target - x_hat_i)

kappa = 0 is no steering, kappa = 1 treats every targeted agent as if
they held x_target. Steering only changes matching_opinions(); the
platform's actual prediction, estimates(), stays unsteered so we can
still measure how accurate it is.
"""

import numpy as np

from sfp.population import MEN, WOMEN


class NoRecommender:
    """Random partners: the 'no algorithm' baseline."""

    def observe(self, x, active, rng):
        pass

    def matching_opinions(self, x):
        return None          # dynamics.py reads None as "pick at random"

    def estimates(self):
        return None


class TrueOpinionMatcher:
    """Matches on true opinions. Same as plain Baumann; our control."""

    def observe(self, x, active, rng):
        pass

    def matching_opinions(self, x):
        return x

    def estimates(self):
        return None


class PredictiveRecommender:
    """Matches on opinions it estimates from noisy, uneven observations."""

    def __init__(self, group, lam=5.0, noise_sd=0.1, window=None,
                 steer_group=None, kappa=0.0, x_target=0.0):
        """
        group       array of 0/1 group labels (only used for steering;
                    the estimator itself never looks at gender)
        lam         shrinkage strength. lam = 0: trust own history fully,
                    no pooling (removes the mechanism, useful as a check)
        noise_sd    standard deviation of the noise on each observation
        window      None = keep all history. A number = old observations
                    fade with that time scale in steps, so the platform
                    tracks recent behaviour (n_i then stays bounded and
                    shrinkage never fully switches off).
        steer_group None, "women", "men" or "all"
        kappa       steering strength, 0..1
        x_target    opinion the platform steers toward
        """
        self.group = np.asarray(group)
        self.N = len(self.group)
        self.lam = lam
        self.noise_sd = noise_sd
        self.decay = None if window is None else 1.0 - 1.0 / window

        self.n = np.zeros(self.N)       # (weighted) number of observations per agent
        self.total = np.zeros(self.N)   # (weighted) sum of observations per agent

        targets = {None: np.zeros(self.N, bool),
                   "women": self.group == WOMEN,
                   "men": self.group == MEN,
                   "all": np.ones(self.N, bool)}
        if steer_group not in targets:
            raise ValueError(f"steer_group must be one of {list(targets)}")
        self.steer_mask = targets[steer_group]
        self.kappa = kappa
        self.x_target = x_target

    def observe(self, x, active, rng):
        """Record a noisy look at every active agent's current opinion."""
        if self.decay is not None:
            self.n *= self.decay
            self.total *= self.decay
        idx = np.flatnonzero(active)
        seen = x[idx] + rng.normal(0.0, self.noise_sd, size=len(idx))
        self.n[idx] += 1
        self.total[idx] += seen

    def pooled_mean(self):
        """Average over all observations (not over users). 0 before any data."""
        n_all = self.n.sum()
        return self.total.sum() / n_all if n_all > 0 else 0.0

    def estimates(self):
        """The platform's prediction x_hat for every agent (no steering)."""
        pooled = self.pooled_mean()
        seen = self.n > 0
        own = np.full(self.N, pooled)
        own[seen] = self.total[seen] / self.n[seen]
        # w = n / (n + lam). Agents never seen get w = 0, i.e. x_hat = pooled.
        # (Written this way so lam = 0 doesn't divide 0 by 0.)
        w = np.zeros(self.N)
        w[seen] = self.n[seen] / (self.n[seen] + self.lam)
        return w * own + (1 - w) * pooled

    def matching_opinions(self, x):
        """What the platform matches on: x_hat, steered for the target group."""
        x_hat = self.estimates()
        if self.kappa > 0:
            s = self.steer_mask
            x_hat[s] += self.kappa * (self.x_target - x_hat[s])
        return x_hat


def make_platform(kind, group, params):
    """Build a platform by name: 'none', 'true' or 'predictive'.

    params is a dict like config.BASELINE; only the recommender and
    steering entries are read.
    """
    if kind == "none":
        return NoRecommender()
    if kind == "true":
        return TrueOpinionMatcher()
    if kind == "predictive":
        return PredictiveRecommender(
            group,
            lam=params["lam"],
            noise_sd=params["noise_sd"],
            window=params["window"],
            steer_group=params["steer_group"],
            kappa=params["kappa"],
            x_target=params["x_target"],
        )
    raise ValueError(f"unknown platform kind: {kind!r} (use 'none', 'true' or 'predictive')")
