"""Figure for the slides: what the synthetic population looks like.

Left:  original opinions theta for each group 
(two normal distributions,
       Delta apart, clipped to [-1, 1]).
Right: activity a on log-log axes. Both groups are drawn from the same
power law; women's activity is then divided by rho, which shifts
their curve to the left by a factor rho.

One population of N = 1000 gives bumpy histograms, so we pool a few
populations (different seeds, same parameters) just for this figure.
Dashed lines are the distributions we sample from, for comparison.

Run from the repo root:  python -m experiments.fig_population
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sfp.config import BASELINE
from sfp.plotting import COLORS
from sfp.population import MEN, WOMEN, make_population

N_POPULATIONS = 10   # pooled only to smooth the histograms


def pooled_population(p, n_pops=N_POPULATIONS):
    """Draw n_pops populations with the baseline parameters and stack them."""
    pops = [make_population(N=p["N"], share_women=p["share_women"], rho=p["rho"],
                            Delta=p["Delta"], sigma=p["sigma"],
                            eps=p["eps"], gamma=p["gamma"], seed=seed)
            for seed in range(n_pops)]
    return {key: np.concatenate([pop[key] for pop in pops]) for key in pops[0]}


def power_law_pdf(a, eps, gamma, scale=1.0):
    """Density of power-law activity after dividing by `scale`.

    The baseline law is f(a) = C a^(-gamma) on [eps, 1]. Dividing every
    activity by `scale` (= rho for women) moves it to [eps/scale, 1/scale]
    and multiplies the density by `scale`.
    """
    C = (gamma - 1) / (eps ** (1 - gamma) - 1)
    pdf = scale * C * (scale * a) ** (-gamma)
    inside = (a >= eps / scale) & (a <= 1 / scale)
    return np.where(inside, pdf, np.nan)


def plot_opinions(ax, pop, p):
    bins = np.linspace(-1, 1, 41)
    xs = np.linspace(-1, 1, 400)
    for g, name, sign in [(MEN, "men", +1), (WOMEN, "women", -1)]:
        theta = pop["theta"][pop["group"] == g]
        mu = sign * p["Delta"] / 2
        ax.hist(theta, bins=bins, density=True, alpha=0.45, color=COLORS[name],
                label=f"{name}  ($\\mu$ = {mu:+.2f})")
        normal = np.exp(-(xs - mu) ** 2 / (2 * p["sigma"] ** 2)) / (p["sigma"] * np.sqrt(2 * np.pi))
        ax.plot(xs, normal, color=COLORS[name], ls="--", lw=1.2)
        ax.axvline(mu, color=COLORS[name], lw=1, alpha=0.8)
    ax.set_xlabel(r"original opinion $\theta_i$")
    ax.set_ylabel("density")
    ax.set_title(rf"$\theta_i \sim \mathcal{{N}}(\pm\Delta/2,\ \sigma)$,  "
                 rf"$\Delta$ = {p['Delta']}, $\sigma$ = {p['sigma']}")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)   # head room for the legend
    ax.legend(frameon=False, loc="upper left", fontsize=9)


def plot_activity(ax, pop, p):
    eps, gamma, rho = p["eps"], p["gamma"], p["rho"]
    bins = np.logspace(np.log10(eps / rho), 0, 35)
    grid = np.logspace(np.log10(eps / rho), 0, 400)
    for g, name, scale in [(MEN, "men", 1.0), (WOMEN, "women", rho)]:
        a = pop["a"][pop["group"] == g]
        dens, edges = np.histogram(a, bins=bins, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        keep = dens > 0
        ax.plot(centers[keep], dens[keep], "o", ms=4, color=COLORS[name],
                label=f"{name}  (mean a = {a.mean():.3f})")
        ax.plot(grid, power_law_pdf(grid, eps, gamma, scale), "--", lw=1.2, color=COLORS[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"activity $a_i$ (chance of being active per step)")
    ax.set_ylabel("density")
    ax.set_title(rf"$F(a) \propto a^{{-\gamma}}$ on $[\varepsilon, 1]$,  women $\div\,\rho$ = {rho}")

    # Arrow showing the shift by rho, drawn at a = 0.1 on the men's curve.
    a0 = 0.1
    y0 = power_law_pdf(np.array([a0]), eps, gamma)[0]
    ax.annotate("", xy=(a0 / rho, y0), xytext=(a0, y0),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2))
    ax.text(np.sqrt(a0 * a0 / rho), y0 / 2.2, rf"$\div\,\rho$", ha="center", va="top", color="0.3")
    ax.legend(frameon=False)


def main():
    p = BASELINE
    pop = pooled_population(p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    plot_opinions(ax1, pop, p)
    plot_activity(ax2, pop, p)
    fig.tight_layout()

    out = Path("figures") / "population.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved {out}")

    # Numbers you may want on the slide.
    a, g = pop["a"], pop["group"]
    share = a[g == WOMEN].sum() / a.sum()
    print(f"mean activity  men {a[g == MEN].mean():.4f}  women {a[g == WOMEN].mean():.4f}  "
          f"ratio {a[g == MEN].mean() / a[g == WOMEN].mean():.2f}")
    print(f"women's expected share of observations: {share:.3f} (NYT: 0.248)")
    print(f"theta clipped at +-1: {np.mean(np.abs(pop['theta']) == 1):.2%} of agents")


if __name__ == "__main__":
    main()