"""The platform: what it knows about users and how it matches them.

3 recommender systems (one per condition in our experiments)

  1. NoRecommender: partners picked at random (no matching at all)
  2. TrueOpinionMatcher: matches on true opinions; a perfect recommender (plain Baumann and control condition)
  3. PredictiveRecommender matches on estimated opinions x_hat, built from noisy observations (introduces bias)

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

