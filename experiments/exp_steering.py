"""Experiment 3: deliberate steering (from the consultation feedback).

Same predictive recommender, but now the platform also nudges the
opinions it matches on toward X_TARGET, either for women only
(targeted) or for everyone (untargeted):

    x_hat_i <- x_hat_i + kappa * (X_TARGET - x_hat_i)

This changes who people get matched with, never their opinions
directly. Any movement in opinions comes from the new contacts.

Questions:
  - Does steering women's matches move women's opinions, and how far?
  - Does targeting one group also move the other one (spillover)?
  - Does the platform's error on women vs their current opinions drop
    while they move away from their original opinions? That is the
    self-fulfilling signature, now produced on purpose.

X_TARGET = +1 is on the men's side (men start at +Delta/2).

Known property of the model (found when first running this): UNTARGETED
steering does (practically) nothing. Pulling everyone's x_hat toward the
same target just shrinks all distances |x_hat_i - x_hat_j| by the same factor
(1 - kappa), and the homophily rule only cares about RELATIVE distances
(the weights get normalised). So the same people get matched as without
steering, up to tiny floating-point differences in the sampling.
Targeted steering works because only women's x_hat move, which
changes who is close to whom. If we want a meaningful untargeted
condition, the platform has to do something else, e.g. boost exposure to
agents whose opinions are near the target. Group decision; the
untargeted rows are kept here so the slides can show the no-effect result.

Output: figures/exp_steering.png and a table.

Run from the repo root:  python -m experiments.exp_steering
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sfp.plotting import band
from sfp.runner import run_condition

SEEDS = range(3)
X_TARGET = 1.0
CONDITIONS = {
    "no steering":           {},
    "targeted, kappa=0.25":  {"steer_group": "women", "kappa": 0.25},
    "targeted, kappa=0.5":   {"steer_group": "women", "kappa": 0.5},
    "untargeted, kappa=0.25": {"steer_group": "all", "kappa": 0.25},
    "untargeted, kappa=0.5": {"steer_group": "all", "kappa": 0.5},
}
FIG = Path(__file__).resolve().parent.parent / "figures" / "exp_steering.png"


def main():
    res = {}
    for name, over in CONDITIONS.items():
        if over:
            over = {**over, "x_target": X_TARGET}
        res[name] = run_condition(over, "predictive", SEEDS, subfolder="steering")

    panels = [("mean_women", "women: mean opinion"),
              ("mean_men", "men: mean opinion"),
              ("err_vs_x_women", "women: error of x_hat vs current x"),
              ("err_vs_theta_women", "women: error of x_hat vs original theta")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, (key, title) in zip(axes, panels):
        for name, r in res.items():
            ls = "--" if name.startswith("untargeted") else "-"
            band(ax, r["times"], r[key], name, ls=ls)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)

    print(f"{'condition':24s} {'mean women':>11s} {'mean men':>9s} "
          f"{'err women vs x':>15s} {'err women vs theta':>19s}")
    for name, r in res.items():
        print(f"{name:24s} {r['mean_women'][:, -1].mean():+11.3f} "
              f"{r['mean_men'][:, -1].mean():+9.3f} "
              f"{r['err_vs_x_women'][:, -1].mean():15.3f} "
              f"{r['err_vs_theta_women'][:, -1].mean():19.3f}")
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
