"""Opinion dynmics, following Baumann et al. (2020) and our extension anchor/stubbornness

Each agent i has opinion x_i
opinion changes: 

    dx_i/dt = -x_i + s*theta_i + K * sum_j A_ij(t) * tanh(alpha * x_j)

  -x_i forgetting: without input, opinions decay
  s*theta_i  anchor to the agent's original opinion theta_i (Friedkin & Johnsen 1990). s = 0 gives plain Baumann.
  K * sum ... social influence from everyone j who reached i this step.
    tanh caps each neighbour's push at K; alpha sets how fast a neighbour reaches that cap.

"""


import numpy as np


def sample_activities(N, rng, eps=0.01, gamma=2.1):
    """Draw N activities from F(a) ~ a^(-gamma) on [eps, 1].

    Inverse-CDF sampling: push uniform random numbers through the inverse
    of the power law's cumulative distribution. Most agents end up near
    eps (rarely active), a few near 1 (active almost every step).
    """
    u = rng.random(N)
    lo = eps ** (1 - gamma)
    return (lo + u * (1 - lo)) ** (1 / (1 - gamma))


def build_network(x_match, active, m, beta, r, rng, dist_floor=1e-6):
    """Build one snapshot of the network, A[i, j] = 1 if j influences i.

    x_match : opinions used to pick partners (true x, or the platform's x_hat)
    active  : boolean array, who is active this step
    beta    : homophily exponent; 0 means partners are picked uniformly
    dist_floor stops 1/0 when two agents have (almost) the same opinion.
    """
    N = len(x_match)
    A = np.zeros((N, N))
    for i in np.flatnonzero(active):
        d = np.abs(x_match[i] - x_match)
        w = np.maximum(d, dist_floor) ** (-beta)
        w[i] = 0.0                          # nobody contacts themselves
        js = rng.choice(N, size=m, replace=False, p=w / w.sum())
        A[js, i] = 1.0                      # i influences each j it reached
        back = js[rng.random(m) < r]
        A[i, back] = 1.0                    # some of them influence i back
    return A


def rk4_step(x, A, K, alpha, s, theta, dt):
    """Advance all opinions by one step dt with the network A held fixed."""
    def f(y):
        return -y + s * theta + K * (A @ np.tanh(alpha * y))
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(N=1000, T=10.0, dt=0.01, K=3.0, alpha=3.0, beta=3.0, r=0.5,
             m=10, eps=0.01, gamma=2.1, s=0.0, x0=None, theta=None, a=None,
             platform=None, seed=0, record_every=10):
    """Run the model and return the recorded trajectory.

    Defaults are Baumann's Fig. 1(c) setup, so simulate() on its own gives
    plain Baumann polarization. For our model, pass the population from
    population.py (x0, theta, a) and a platform from recommender.py.
    Easier still: use runner.run(), which does all of that for you.

    x0       starting opinions (default: evenly spaced on [-1, 1], like Baumann)
    theta    original opinions for the anchor term (default: same as x0)
    a        activities (default: fresh draw from the power law)
    platform None = match on true opinions (plain Baumann)

    Returns a dict:
      times   (R,)    time of each recorded snapshot
      x       (R, N)  opinions at each snapshot
      x_hat   (R, N)  platform estimates at each snapshot (None if no platform
                      or the platform makes no estimates)
      a       (N,)    activities used
      n_active (N,)   how often each agent was active during the run
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, N) if x0 is None else np.asarray(x0, float).copy()
    theta = x.copy() if theta is None else np.asarray(theta, float)
    a = sample_activities(N, rng, eps, gamma) if a is None else np.asarray(a)

    def snapshot_estimate():
        est = None if platform is None else platform.estimates()
        return None if est is None else est.copy()

    xs, xhats = [x.copy()], [snapshot_estimate()]
    n_active = np.zeros(N, dtype=int)

    steps = int(round(T / dt))
    for t in range(1, steps + 1):
        active = rng.random(N) < a
        n_active += active

        # Who gets matched with whom. With a platform, it first sees this
        # step's activity, then says which opinions to compare.
        if platform is None:
            x_match, b = x, beta
        else:
            platform.observe(x, active, rng)
            x_match = platform.matching_opinions(x)
            if x_match is None:              # platform matches at random
                x_match, b = x, 0.0
            else:
                b = beta
        A = build_network(x_match, active, m, b, r, rng)

        # Influence always runs on the true opinions.
        x = rk4_step(x, A, K, alpha, s, theta, dt)

        if t % record_every == 0:
            xs.append(x.copy())
            xhats.append(snapshot_estimate())

    has_est = all(e is not None for e in xhats)
    return {
        "times": np.arange(len(xs)) * dt * record_every,
        "x": np.array(xs),
        "x_hat": np.array(xhats) if has_est else None,
        "a": a,
        "n_active": n_active,
    }
