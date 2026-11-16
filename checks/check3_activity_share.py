"""Check 3: does the population reproduce the NYT activity gap?

With 44% women and rho = 2.4, women should produce about 25% of all
activations (= observations the platform gets), like the 24.8% of NYT
comments written by women in Pierson (2015).

Pure counting, no opinion dynamics: draw the population, then simulate
who is active over 1000 steps.

Run from the repo root:  python -m checks.check3_activity_share
"""

import numpy as np

from sfp.config import BASELINE
from sfp.population import MEN, WOMEN, make_population


def women_share_of_activations(pop, steps=1000, seed=0):
    """Fraction of all activations over `steps` steps that came from women."""
    rng = np.random.default_rng(seed)
    a, group = pop["a"], pop["group"]
    counts = np.zeros(len(a))
    for _ in range(steps):
        counts += rng.random(len(a)) < a
    return counts[group == WOMEN].sum() / counts.sum()


def main():
    p = BASELINE
    shares = []
    for seed in range(20):
        pop = make_population(N=p["N"], share_women=p["share_women"], rho=p["rho"],
                              eps=p["eps"], gamma=p["gamma"], seed=seed)
        shares.append(women_share_of_activations(pop, seed=seed))
    shares = np.array(shares)
    print(f"women's share of observations: {shares.mean():.3f} "
          f"(sd {shares.std():.3f} over 20 populations); target about 0.25")

    pop = make_population(rho=p["rho"], seed=0)
    a, g = pop["a"], pop["group"]
    print(f"mean activity  men {a[g == MEN].mean():.4f}  women {a[g == WOMEN].mean():.4f}  "
          f"ratio {a[g == MEN].mean() / a[g == WOMEN].mean():.2f} (rho = {p['rho']})")


if __name__ == "__main__":
    main()
