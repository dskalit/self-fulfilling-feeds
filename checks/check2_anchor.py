"""
Check 2: what does the anchor term s do to Baumann's regimes?
"""
import matplotlib.pyplot as plt
import numpy as np

from sfp.config import BASELINE
from sfp.dynamics import simulate
from sfp.metrics import extremity, share_positive
from sfp.runner import RESULTS_DIR

REGIMES = {
    "consensus (a=0.05, b=2)": dict(alpha=0.05, beta=2),
    "radicalization (a=3, b=0)": dict(alpha=3, beta=0),
    "polarization (a=3, b=3)": dict(alpha=3, beta=3),
}
S_VALUES = [0.0, 0.5, 1.0, 2.0]


def describe(final):
    """One line per run: mean opinion, share on the positive side,
    extremity (mean |x|) and the most extreme agent."""
    return (f"|mean| {abs(final.mean()):5.2f}  share+ {share_positive(final):.2f}  "
            f"mean|x| {extremity(final):5.2f}  max|x| {np.abs(final).max():5.1f}")


def part_a():
    """
    Baumann's regime (same as check 1)
    but with s = 0, 0.5, 1, 2
    """
    print("Part A: Baumann's regimes with the anchor term")
    fig, axes = plt.subplots(len(S_VALUES), 3, figsize=(13, 2.6 * len(S_VALUES)),
                             sharex=True)
    for row, s in enumerate(S_VALUES):
        for col, (name, p) in enumerate(REGIMES.items()):
            res = simulate(K=3, r=0.5, s=s, **p, seed=1)
            x, final = res["x"], res["x"][-1]
            colors = np.where(final > 0, "tab:blue", "tab:red")
            ax = axes[row, col]
            for i in range(0, x.shape[1], 5):
                ax.plot(res["times"], x[:, i], color=colors[i], lw=0.4, alpha=0.6)
            if row == 0:
                ax.set_title(name, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"s = {s}")
            print(f"  s={s:<4} {name:28s} {describe(final)}")
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.tight_layout()
    out = RESULTS_DIR / "check2_anchor_regimes.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"  saved {out}\n")


def part_b():
    """
    Sweep alpha and beta at s = whatever is specified in config.py to find baseline
    """
    s = BASELINE["s"]
    print(f"Part B: alpha x beta grid at s = {s} (K = {BASELINE['K']})")
    for alpha in [0.5, 1.0, 2.0, 3.0]:
        for beta in [0.0, 1.0, 3.0]:
            res = simulate(K=BASELINE["K"], r=BASELINE["r"], s=s,
                           alpha=alpha, beta=beta, seed=1)
            print(f"  alpha={alpha:<4} beta={beta:<4} {describe(res['x'][-1])}")

def part_c():
    """Sweep K, alpha and s at beta = 3 (polarization-like homophily)."""
    print("Part C: K x alpha x s at beta = 3")
    for K in [0.5, 1.0, 2.0, 3.0]:
        for alpha in [0.5, 1.0, 3.0]:
            for s in [1.0, 2.0, 4.0]:
                res = simulate(K=K, r=BASELINE["r"], s=s,
                               alpha=alpha, beta=3, seed=1)
                print(f"  K={K:<4} alpha={alpha:<4} s={s:<4} {describe(res['x'][-1])}")

if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
