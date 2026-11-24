"""Default parameters for every run, kept in one place.

Checks and experiments start from BASELINE and only override what they
change, e.g. run({"rho": 3.0}). 

Updating params based on check 2
"""

BASELINE = {
    # --- Simulation ---------------------------------------------------
    "N": 1000,          # number of agents (Baumann: 1000)
    "T": 10.0,          # total simulated time (Baumann: 10)
    "dt": 0.01,         # time step; the network is redrawn every step (Baumann: 0.01)
    "record_every": 10,  # store opinions every 10 steps (= every 0.1 time units)

    # --- Opinion dynamics (Baumann et al. 2020) -----------------------
    "K": 1.5,           # strength of social influence
    "alpha": 3.0,       # controversialness: how quickly influence saturates
    "beta": 3.0,        # homophily: how strongly matching favours similar opinions
    "r": 0.5,           # chance a contact also influences the active agent back
    "m": 10,            # contacts per activation
    "eps": 0.01,        # minimum activity in the power law
    "gamma": 2.1,       # power-law exponent of activity

    # --- Our extension: anchor to original opinion --------------------
    # dx/dt = -x + s*theta + social influence. With no social input an
    # agent relaxes to x = s*theta, so s = 1 means "you drift back to what
    # you originally thought". s = 0 is plain Baumann.
    "s": 1.0,

    # --- Population (population.py) -----------------------------------
    "share_women": 0.44,  # NYT readership share (Pierson 2015)
    "rho": 2.4,           # men's activity / women's activity (NYT comments, Pierson 2015)
    "Delta": 0.5,         # gap between group means of theta (men +Delta/2, women -Delta/2).
                          # Needs to be > 0 to see one group pulled toward the other
                          # (check 4 runs Delta = 0 on purpose). PROVISIONAL value.
    "sigma": 0.3,         # spread of theta within each group

    # --- Recommender (recommender.py) ---------------------------------
    "noise_sd": 0.1,    # noise on each observation of an agent's opinion
    "lam": 5.0,         # shrinkage strength: how many observations before the
                        # platform trusts an agent's own history over the pooled mean
    "window": None,     # None = remember all observations; a number = forget old
                        # observations with that time scale (in steps)

    # --- Steering (only used in the steering experiment) --------------
    "steer_group": None,  # None, "women", "men" or "all"
    "kappa": 0.0,         # how far the platform pulls x_hat toward x_target (0..1)
    "x_target": 0.0,      # the opinion the platform steers toward
}
