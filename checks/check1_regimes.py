"""Check 1: does our code reproduce Baumann et al. Fig. 1?

Plain Baumann (s = 0, no platform, no groups), K = 3, r = 0.5, opinions
starting evenly spaced on [-1, 1]. 

    Parameters                   Expected figure
  (a) alpha = 0.05, beta = 2  -> consensus: everyone decays to 0
  (b) alpha = 3,    beta = 0  -> radicalization: everyone ends on ONE side
                                 (which side is random)
  (c) alpha = 3,    beta = 3  -> polarization: two camps, roughly 50/50


"""

import matplotlib.pyplot as plt
import numpy as np

from sfp.dynamics import simulate
from sfp.runner import RESULTS_DIR

REGIMES = {
    "(a) consensus: alpha=0.05, beta=2": dict(alpha=0.05, beta=2),
    "(b) radicalization: alpha=3, beta=0": dict(alpha=3, beta=0),
    "(c) polarization: alpha=3, beta=3": dict(alpha=3, beta=3),
}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (title, p) in zip(axes, REGIMES.items()):
        res = simulate(K=3, r=0.5, s=0, **p, seed=1)
        times, x = res["times"], res["x"]
        final = x[-1]

        # Colour each agent by the side it ends up on, like Baumann.
        colors = np.where(final > 0, "tab:blue", "tab:red")
        for i in range(0, x.shape[1], 5):          # every 5th agent keeps it readable
            ax.plot(times, x[:, i], color=colors[i], lw=0.4, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t")

        print(f"{title:38s} |mean| = {abs(final.mean()):.3f}   "
              f"share positive = {np.mean(final > 0):.2f}   "
              f"range = [{final.min():.2f}, {final.max():.2f}]")

    axes[0].set_ylabel("opinion x_i(t)")
    fig.tight_layout()
    out = RESULTS_DIR / "check1_regimes.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
