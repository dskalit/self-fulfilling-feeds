"""Check 4: is the recommender's bias coming from the right place?

The pooled mean pulls under-observed agents toward the average of all
observations. So:

Delta = 0  
(groups start with the same opinions): the pooled mean is
the shared centre, so there's nothing to drift toward:
the group gap should stay about 0 under every platform.
(We also expected the spread to shrink under 'predictive'.
At the current baseline that effect is tiny, because
radicalization spreads opinions out far more than shrinkage
pulls them in.)

Delta > 0  
(men start at +Delta/2, women at -Delta/2): the pooled mean
sits near the men, so women should drift toward the men
(positive drift_women) more under 'predictive' than 'true'.

lam = 0    
no pooling at all: 'predictive' should behave like 'true'
apart from observation noise.

Averaged over several seeds (single runs are noisy)

Run from the repo root:  python -m checks.check4_no_drift
"""

import numpy as np

from sfp.metrics import summarize
from sfp.runner import run

SEEDS = range(5)
CASES = [
    ("Delta=0,   true",        {"Delta": 0.0},             "true"),
    ("Delta=0,   predictive",  {"Delta": 0.0},             "predictive"),
    ("Delta=0.5, true",        {"Delta": 0.5},             "true"),
    ("Delta=0.5, predictive",  {"Delta": 0.5},             "predictive"),
    ("Delta=0.5, pred. lam=0", {"Delta": 0.5, "lam": 0.0}, "predictive"),
]


def main():
    cols = ["gap start", "gap end", "drift men", "drift women",
            "spread men", "spread wom."]
    print(f"{'case':24s} " + "".join(f"{c:>12s} " for c in cols))
    print("(each cell: mean ± sd over seeds, at the end of the run unless it says start)")
    for label, overrides, platform in CASES:
        rows = []
        for seed in SEEDS:
            m = summarize(run(overrides, platform, seed))
            rows.append([m["gap"][0], m["gap"][-1], m["drift_men"][-1],
                         m["drift_women"][-1], m["spread_men"][-1],
                         m["spread_women"][-1]])
        mean, sd = np.mean(rows, axis=0), np.std(rows, axis=0)
        cells = "".join(f"{f'{mu:+.2f}±{s:.2f}':>12s} " for mu, s in zip(mean, sd))
        print(f"{label:24s} {cells}")


if __name__ == "__main__":
    main()