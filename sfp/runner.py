"""Running the code: build a population and a platform, run the model, save results.
"""

import json
from pathlib import Path

import numpy as np

from sfp.config import BASELINE
from sfp.dynamics import simulate
from sfp.metrics import summarize
from sfp.population import make_population
from sfp.recommender import make_platform

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def full_params(overrides=None):
    """BASELINE with some entries replaced. Catches typos in parameter names."""
    overrides = overrides or {}
    unknown = set(overrides) - set(BASELINE)
    if unknown:
        raise KeyError(f"unknown parameter(s): {sorted(unknown)}")
    return {**BASELINE, **overrides}


def run(overrides=None, platform="predictive", seed=0):
    """One simulation. Returns simulate()'s dict plus 'group', 'theta',
    'params', 'platform' and 'seed'.

    The same seed gives the same population across platforms, so 'true'
    vs 'predictive' with seed=3 compares the same people under two
    different recommenders.
    """
    p = full_params(overrides)
    pop = make_population(N=p["N"], share_women=p["share_women"], rho=p["rho"],
                          Delta=p["Delta"], sigma=p["sigma"], eps=p["eps"],
                          gamma=p["gamma"], seed=seed)
    plat = make_platform(platform, pop["group"], p)

    res = simulate(N=p["N"], T=p["T"], dt=p["dt"], K=p["K"], alpha=p["alpha"],
                   beta=p["beta"], r=p["r"], m=p["m"], s=p["s"],
                   x0=pop["theta"], theta=pop["theta"], a=pop["a"],
                   platform=plat, seed=seed + 10_000,   # separate stream from population
                   record_every=p["record_every"])
    res.update(group=pop["group"], theta=pop["theta"], params=p,
               platform=platform, seed=seed)
    return res


def run_name(overrides, platform, seed):
    """File name from the parameters that differ from BASELINE,
    e.g. 'predictive__lam=10__rho=3.0__seed2.npz'."""
    parts = [platform] + [f"{k}={v}" for k, v in sorted((overrides or {}).items())]
    return "__".join(parts) + f"__seed{seed}.npz"


def save_run(res, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in res.items()
              if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays,
                        params=json.dumps(res["params"]),
                        platform=res["platform"], seed=res["seed"])


def load_run(path):
    d = np.load(path, allow_pickle=False)
    res = {k: d[k] for k in d.files if k not in ("params", "platform", "seed")}
    res["params"] = json.loads(str(d["params"]))
    res["platform"] = str(d["platform"])
    res["seed"] = int(d["seed"])
    res.setdefault("x_hat", None)     # not saved when the platform had none
    return res


def run_cached(overrides=None, platform="predictive", seed=0, subfolder=""):
    """Like run(), but loads results/<subfolder>/<name>.npz if it exists."""
    path = RESULTS_DIR / subfolder / run_name(overrides, platform, seed)
    if path.exists():
        return load_run(path)
    res = run(overrides, platform, seed)
    save_run(res, path)
    return res


def run_condition(overrides=None, platform="predictive", seeds=range(5),
                  subfolder=""):
    """Run (or load) one condition for several seeds and summarize each run.

    Returns a dict of metric -> array of shape (n_seeds, n_snapshots),
    plus 'times'. Average over axis 0 to get the mean over seeds.
    """
    seeds = list(seeds)
    summaries = [summarize(run_cached(overrides, platform, seed, subfolder))
                 for seed in seeds]
    out = {k: np.array([s[k] for s in summaries]) for k in summaries[0]}
    out["times"] = run_cached(overrides, platform, seeds[0], subfolder)["times"]
    return out
