"""Small plotting helpers shared by the experiment scripts, so all figures
use the same colours and style."""

import numpy as np

COLORS = {"men": "tab:blue", "women": "tab:orange"}


def band(ax, times, runs, label=None, color=None, ls="-"):
    """Plot the mean over seeds as a line and +-1 sd as a shaded band.

    runs: array of shape (n_seeds, n_snapshots), e.g. run_condition(...)["mean_women"]
    """
    mean, sd = np.nanmean(runs, axis=0), np.nanstd(runs, axis=0)
    line, = ax.plot(times, mean, label=label, color=color, ls=ls, lw=1.6)
    ax.fill_between(times, mean - sd, mean + sd, color=line.get_color(), alpha=0.2, lw=0)
    return line
