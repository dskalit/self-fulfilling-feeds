"""Figures used in the Experiment 2 and 3 slides (presentation, 30 Nov).

Simpler, presentation versions of the plots made by exp_unintentional_bias.py
and exp_steering.py: no parameter symbols, lines labelled directly, so an
audience that has not seen the model can read them. Same runs, same seeds
(3), same cached results in results/unintentional_bias/ and
results/steering/, so the numbers match the experiment scripts.

  figures/slides/exp2_men_vs_women.png
      Average opinion of men and women over time at the NYT activity gap
      (rho = 2.4): perfect recommender ('true') vs the guessing platform
      ('predictive', lam = 5). Slide "Men move toward women, not the other
      way round".
  figures/slides/exp3_women_steering.png
      Women's average opinion over time with no steering, steering women
      (kappa = 0.25 'weak', 0.5 'strong') and steering everyone
      (kappa = 0.5, dotted; lies on top of no steering). Slide "Steering
      women moves women".
  figures/slides/exp3_error_now_vs_start.png
      The platform's error about women, against their current opinions
      (what it can check) and against their original opinions (what it
      cannot), no steering vs strong steering. Slide "The platform grades
      its own homework".

Numbers quoted on the slides (from the same runs):
  Exp 2: men end 0.21 closer to women than with a perfect recommender
         (+0.557 vs +0.763); women -0.493 vs -0.501.
  Exp 3: strong targeted steering moves women 0.39 toward men
         (-0.099 vs -0.493), men 0.14 toward women; steering everyone ~0.
         Error vs current opinions drops to 0.24 (no steering peaks at 0.69);
         error vs original opinions ends near 0.48 either way.

Run from the repo root:  python -m experiments.slide_figures
(Delete results/ first if the model code changed; the cache does not notice.)
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # much faster for N = 1000
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sfp.runner import run_condition

OUT = Path(__file__).resolve().parent.parent / "figures" / "slides"
SEEDS = range(3)
X_TARGET = 1.0
BLUE, ORANGE, GREEN, GREY, PURPLE = "#2a6fdb", "#f28c28", "#2ca02c", "#999999", "#7b3fe4"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def line(ax, t, runs, color, ls="-", lw=2.4):
    """Mean over seeds as a line, +-1 sd as a band. Returns the mean."""
    m, s = runs.mean(0), runs.std(0)
    ax.plot(t, m, color=color, ls=ls, lw=lw)
    ax.fill_between(t, m - s, m + s, color=color, alpha=0.12, lw=0)
    return m


def start_end_axis(ax, xmax):
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels(["start", "", "end"])


def exp2_men_vs_women():
    true = run_condition({"rho": 2.4}, "true", SEEDS, "unintentional_bias")
    pred = run_condition({"rho": 2.4, "lam": 5.0}, "predictive", SEEDS, "unintentional_bias")
    t = true["times"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    a = line(ax, t, true["mean_men"], BLUE, "-")
    b = line(ax, t, pred["mean_men"], BLUE, "--")
    c = line(ax, t, true["mean_women"], ORANGE, "-")
    d = line(ax, t, pred["mean_women"], ORANGE, "--")
    ax.text(10.2, a[-1], "men, perfect recommender", color=BLUE, va="center", fontsize=11)
    ax.text(10.2, b[-1], "men, guessing platform", color=BLUE, va="center", fontsize=11, fontweight="bold")
    ax.text(10.2, (c[-1] + d[-1]) / 2, "women, both platforms", color=ORANGE, va="center",
            fontsize=11, fontweight="bold")
    ax.annotate("", xy=(9.6, b[-1] + 0.02), xytext=(9.6, a[-1] - 0.02),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.axhline(0, color=GREY, lw=0.8)
    start_end_axis(ax, 15.5)
    ax.set_ylabel("average opinion")
    ax.set_yticks([-0.5, 0, 0.5], ["−0.5", "0", "+0.5"])
    fig.savefig(OUT / "exp2_men_vs_women.png", dpi=220)
    print(f"exp2: men {a[-1]:+.3f} (perfect) vs {b[-1]:+.3f} (guessing); "
          f"women {c[-1]:+.3f} vs {d[-1]:+.3f}")


def steer(group, kappa):
    return {"steer_group": group, "kappa": kappa, "x_target": X_TARGET}


def exp3_women_steering():
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    for name, over, col in [("no steering / steer everyone", {}, GREY),
                            ("steer women, weak", steer("women", 0.25), ORANGE),
                            ("steer women, strong", steer("women", 0.5), GREEN)]:
        r = run_condition(over, "predictive", SEEDS, "steering")
        m = line(ax, r["times"], r["mean_women"], col)
        ax.text(10.2, m[-1], name, color=col, va="center", fontsize=11, fontweight="bold")
        print(f"exp3: {name:28s} women end {m[-1]:+.3f}")
    everyone = run_condition(steer("all", 0.5), "predictive", SEEDS, "steering")
    m = line(ax, everyone["times"], everyone["mean_women"], PURPLE, ":", 2.0)
    print(f"exp3: {'steer everyone, strong':28s} women end {m[-1]:+.3f}")
    ax.axhline(0, color=GREY, lw=0.8)
    start_end_axis(ax, 15.8)
    ax.set_ylabel("women's average opinion")
    ax.set_yticks([-0.5, -0.25, 0], ["−0.5", "−0.25", "0"])
    fig.savefig(OUT / "exp3_women_steering.png", dpi=220)


def exp3_error_now_vs_start():
    none = run_condition({}, "predictive", SEEDS, "steering")
    strong = run_condition(steer("women", 0.5), "predictive", SEEDS, "steering")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, layout="constrained")
    panels = [(axes[0], "err_vs_x_women", "compared with what women think NOW\n(what the platform can check)"),
              (axes[1], "err_vs_theta_women", "compared with what women thought AT THE START\n(what it cannot check)")]
    for ax, key, title in panels:
        a = line(ax, none["times"], none[key], GREY)
        b = line(ax, strong["times"], strong[key], GREEN)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([0, 5, 10], ["start", "", "end"])
        print(f"exp3 {key}: no steering max {a.max():.2f} end {a[-1]:.2f}; "
              f"strong steering min {b.min():.2f} end {b[-1]:.2f}")
    axes[0].set_ylabel("how wrong the platform's\nguess about women is")
    axes[0].text(2.3, 0.72, "no steering", color=GREY, fontsize=11, fontweight="bold")
    axes[0].text(3.5, 0.2, "strong steering", color=GREEN, fontsize=11, fontweight="bold")
    fig.savefig(OUT / "exp3_error_now_vs_start.png", dpi=220)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exp2_men_vs_women()
    exp3_women_steering()
    exp3_error_now_vs_start()
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()
