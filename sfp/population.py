"""Synthetic population: who is in which group, what they originally think,
and how active they are.

We don't use real user data. Instead each agent gets:
  group  0 = men (more active), 1 = women (less active)
  theta  original opinion, drawn from a normal distribution per group:
         men around +Delta/2, women around -Delta/2, spread sigma,
         clipped to [-1, 1]. Delta = 0 means both groups start out with
         the same opinions.
  a      activity, from Baumann's power law, with women's activity
         divided by rho. rho = 2.4 matches the NYT comment section
         (Pierson 2015): women are 44% of readers but write about 25%
         of comments.

Design note (check 3): an earlier version divided by rho and then
clipped women's activity back up to eps. Because most agents sit near
eps, that lifted 63% of women back to eps and shrank the real gap from
2.4 to about 2.15 (women got 27% of observations instead of 25%). Now we
only divide, so women's minimum activity is eps/rho and the gap really is rho.
"""

import numpy as np

from sfp.dynamics import sample_activities

MEN, WOMEN = 0, 1


def make_population(N=1000, share_women=0.44, rho=2.4, Delta=0.0, sigma=0.3,
                    eps=0.01, gamma=2.1, seed=0):
    """Return a dict with arrays 'group', 'theta' and 'a', each of length N."""
    rng = np.random.default_rng(seed)

    # Exactly round(share_women * N) women, at random positions.
    group = np.full(N, MEN)
    n_women = int(round(share_women * N))
    group[rng.choice(N, size=n_women, replace=False)] = WOMEN

    # Original opinions: same spread, means Delta apart.
    mu = np.where(group == MEN, Delta / 2, -Delta / 2)
    theta = np.clip(rng.normal(mu, sigma), -1, 1)

    # Activity: same distribution for everyone, then scale women down.
    a = sample_activities(N, rng, eps, gamma)
    a = np.where(group == WOMEN, a / rho, a)

    return {"group": group, "theta": theta, "a": a}
