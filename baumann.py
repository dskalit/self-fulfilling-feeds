"""
Core opinion dynamics from Baumann et al. (2020), PRL 124, 048301.

    dx_i/dt = -x_i + s*theta_i + K * sum_j A_ij(t) tanh(alpha * x_j)

s = 0 is plain Baumann. The s*theta_i anchor term is our extension
(Friedkin & Johnsen 1990); it's here already so later code doesn't
need a second copy of the integrator.

Temporal network (activity-driven, from the paper's supplement):
  1) each agent i is active with probability a_i
  2) an active i influences m distinct agents j, chosen with
     p_ij proportional to |x_i - x_j|^(-beta)        -> A[j, i] = 1
  3) with probability r the link is reciprocal       -> A[i, j] = 1
  4) integrate one RK4 step (dt = 0.01) with A fixed
  5) delete A
"""

import numpy as np


def sample_activities(N, rng, eps=0.01, gamma=2.1):
    """Inverse-CDF sample from F(a) ~ a^-gamma on [eps, 1]."""
    u = rng.random(N)
    lo = eps ** (1 - gamma)
    return (lo + u * (1 - lo)) ** (1 / (1 - gamma))


def build_network(x_match, active, m, beta, r, rng, dist_floor=1e-6):
    """One snapshot of the temporal adjacency matrix.

    x_match: the opinions the contact choice is based on. In plain
    Baumann this is the true x. Later the platform will pass x_hat
    (predicted opinions) here instead, while influence still uses x.
    """
    N = len(x_match)
    A = np.zeros((N, N))
    for i in np.flatnonzero(active):
        d = np.abs(x_match[i] - x_match)
        d[i] = np.inf                       # no self-contact
        w = np.maximum(d, dist_floor) ** (-beta)
        w[i] = 0.0
        js = rng.choice(N, size=m, replace=False, p=w / w.sum())
        A[js, i] = 1.0                      # i influences j
        recip = js[rng.random(m) < r]
        A[i, recip] = 1.0                   # j influences i back
    return A


def rk4_step(x, A, K, alpha, s, theta, dt):
    def f(y):
        return -y + s * theta + K * (A @ np.tanh(alpha * y))
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(N=1000, T=10.0, dt=0.01, K=3.0, alpha=3.0, beta=3.0, r=0.5,
             m=10, eps=0.01, gamma=2.1, s=0.0, x0=None, theta=None,
             a=None, seed=0, record_every=10):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, N) if x0 is None else np.asarray(x0, float).copy()
    theta = x.copy() if theta is None else np.asarray(theta, float)
    a = sample_activities(N, rng, eps, gamma) if a is None else a

    steps = int(round(T / dt))
    traj = [x.copy()]
    for t in range(1, steps + 1):
        active = rng.random(N) < a
        A = build_network(x, active, m, beta, r, rng)
        x = rk4_step(x, A, K, alpha, s, theta, dt)
        if t % record_every == 0:
            traj.append(x.copy())
    times = np.arange(len(traj)) * dt * record_every
    return times, np.array(traj), a


# Check 1: reproduce Baumann Fig. 1 (K = 3, r = 0.5 in all three panels)
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    regimes = {
        "(a) consensus: alpha=0.05, beta=2": dict(alpha=0.05, beta=2),
        "(b) radicalization: alpha=3, beta=0": dict(alpha=3, beta=0),
        "(c) polarization: alpha=3, beta=3": dict(alpha=3, beta=3),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (title, p) in zip(axes, regimes.items()):
        times, traj, _ = simulate(**p, seed=1)
        final = traj[-1]
        colors = np.where(final > 0, "tab:blue", "tab:red")
        for i in range(0, traj.shape[1], 5):  # every 5th agent for speed
            ax.plot(times, traj[:, i], color=colors[i], lw=0.4, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t")
        print(f"{title:40s} mean={final.mean():+.3f}  "
              f"|mean|={abs(final.mean()):.3f}  "
              f"share positive={np.mean(final > 0):.2f}  "
              f"range=[{final.min():.2f}, {final.max():.2f}]")
    axes[0].set_ylabel("x_i(t)")
    fig.tight_layout()
    fig.savefig("check1_baumann_regimes.png", dpi=150)
    print("saved check1_baumann_regimes.png")