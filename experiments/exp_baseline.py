"""Experiment 1: the same population under three platforms.

  none        random partners, no recommender at all
  true        matches on true opinions (perfect recommender, control)
  predictive  matches on estimated opinions x_hat (our mechanism)

All at the baseline parameters in sfp/config.py, same seeds for every
platform, so differences come from the platform and nothing else.

What to look at:
  - group means over time: do men and women end up closer together
    under 'predictive' than under 'true'? Who moved?
  - prediction error of the predictive platform, against current
    opinions x and against original opinions theta. If error vs x falls
    while opinions move away from theta, the platform is getting "more
    accurate" by changing people: the self-fulfilling signature.

Output: figures/exp_baseline.png and a table of final values.

Run from the repo root:  python -m experiments.exp_baseline
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sfp.plotting import COLORS, band
from sfp.runner import run_condition

SEEDS = range(5)
PLATFORMS = ["none", "true", "predictive"]
FIG = Path(__file__).resolve().parent.parent / "figures" / "exp_baseline.png"


def main():
    res = {pf: run_condition({}, pf, SEEDS, subfolder="baseline") for pf in PLATFORMS}

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, pf in zip(axes, PLATFORMS):
        r = res[pf]
        band(ax, r["times"], r["mean_men"], "men", COLORS["men"])
        band(ax, r["times"], r["mean_women"], "women", COLORS["women"])
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"platform: {pf}")
        ax.set_xlabel("t")
    axes[0].set_ylabel("mean opinion")
    axes[0].legend()

    r = res["predictive"]
    ax = axes[3]
    band(ax, r["times"], r["err_vs_x"], "x_hat vs current x")
    band(ax, r["times"], r["err_vs_theta"], "x_hat vs original theta")
    ax.set_title("predictive: prediction error")
    ax.set_xlabel("t")
    ax.legend()

    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)

    print(f"{'platform':11s} {'mean men':>9s} {'mean women':>11s} {'gap':>7s} "
          f"{'extremity':>10s} {'share +':>8s}")
    for pf in PLATFORMS:
        r = res[pf]
        vals = [r[k][:, -1].mean() for k in
                ["mean_men", "mean_women", "gap", "extremity", "share_positive"]]
        print(f"{pf:11s} {vals[0]:+9.3f} {vals[1]:+11.3f} {vals[2]:+7.3f} "
              f"{vals[3]:10.3f} {vals[4]:8.2f}")
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
