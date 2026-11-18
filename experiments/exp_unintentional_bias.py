"""Experiment 2: unintentional bias. How does the effect of the predictive
recommender depend on the activity gap (rho) and on how hard it pools
(lam)?

For every (rho, lam) pair we compare 'predictive' against 'true' on the
same seeds, so each population is its own control:

    women's pull = mean_women(predictive) - mean_women(true)
    men's pull   = mean_men(true) - mean_men(predictive)

Men start on the positive side (+Delta/2), women on the negative side,
so with these signs a positive pull means "moved toward the other
group". If the activity gap is what drives the bias, women's pull
should grow with rho, and the difference (women's pull - men's pull)
should turn positive as rho grows. At rho = 1 there's no activity gap,
so any pull there comes from pooling alone. At lam = 0 there's no
pooling, so both pulls should be about 0.

Output: figures/exp_unintentional_bias.png (three heatmaps) and a table.

Run from the repo root:
    python -m experiments.exp_unintentional_bias          full sweep (slow)
    python -m experiments.exp_unintentional_bias --quick  2 x 2 grid, 1 seed
Runs are cached in results/unintentional_bias/, so a second run is fast.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sfp.runner import run_condition

FIG = Path(__file__).resolve().parent.parent / "figures" / "exp_unintentional_bias.png"


def main(quick=False):
    rhos = [1.0, 2.4] if quick else [1.0, 1.5, 2.0, 2.4, 3.0]
    lams = [0.0, 5.0] if quick else [0.0, 2.0, 5.0, 10.0, 20.0]
    seeds = range(1) if quick else range(3)
    sub = "unintentional_bias"

    women_pull = np.zeros((len(lams), len(rhos)))
    men_pull = np.zeros_like(women_pull)
    for j, rho in enumerate(rhos):
        # lam doesn't matter for 'true', so one control per rho
        true = run_condition({"rho": rho}, "true", seeds, sub)
        for i, lam in enumerate(lams):
            pred = run_condition({"rho": rho, "lam": lam}, "predictive", seeds, sub)
            women_pull[i, j] = (pred["mean_women"][:, -1] - true["mean_women"][:, -1]).mean()
            men_pull[i, j] = (true["mean_men"][:, -1] - pred["mean_men"][:, -1]).mean()
            print(f"rho={rho:<4} lam={lam:<5} women's pull {women_pull[i, j]:+.3f}   "
                  f"men's pull {men_pull[i, j]:+.3f}")

    panels = [("women's pull toward men", women_pull),
              ("men's pull toward women", men_pull),
              ("women's pull - men's pull", women_pull - men_pull)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, data) in zip(axes, panels):
        lim = np.abs(data).max() or 1.0
        im = ax.imshow(data, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       aspect="auto")
        ax.set_xticks(range(len(rhos)), [str(r) for r in rhos])
        ax.set_yticks(range(len(lams)), [str(l) for l in lams])
        ax.set_xlabel("rho (activity gap)")
        ax.set_ylabel("lam (pooling strength)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"saved {FIG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small grid, one seed")
    main(parser.parse_args().quick)
