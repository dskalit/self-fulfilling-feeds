# self-fulfilling-feeds

Group project for *Complex Social Systems: Modeling Agents, Learning, and Games* (ETH Zürich, HS 2026).

Team members: 
Daria Skalitzky (dskalit)
Zihan Zhou (zihzhou)
Vania Halim (vhalim)

**When Predictions Shape Preferences: self-fulfilling prophecies and gender bias in recommender systems**

## The idea

A recommender has to guess what each user thinks, and then shows them content and people based on that guess. If the guess is worse for some users, and what people are shown changes what they think, then the guess can end up making itself true. Khosrowi et al. (2025) use gender as an example of this kind of self-fulfilling prediction.

Our question is a general version of that: can the more active group on a platform end up shaping the opinions of a quieter group, through a recommender that never looks at group membership? Gender is our application. In the NYT comment section, women are 44% of readers but write only about 25% of the comments (Pierson 2015), so a platform simply sees much less of them.

# Model

config.BASELINE            the dict of parameter names and defaults (copy it as-is)
make_population(...)   ->  {"group": (N,), "theta": (N,), "a": (N,)}
platform object            observe(x, active, rng); matching_opinions(x) -> array or None;
                           estimates() -> array or None
simulate(...)          ->  {"times", "x" (R,N), "x_hat" (R,N) or None, "a", "n_active"}
summarize(result)      ->  dict of metric name -> array over time


## References

Baumann, F., Lorenz-Spreen, P., Sokolov, I. M., & Starnini, M. (2020). Modeling echo chambers and polarization dynamics in social networks. Physical Review Letters, 124(4), 048301. https://doi.org/10.1103/PhysRevLett.124.048301

Khosrowi, D., Ahlers, M., & van Basshuysen, P. (2025). When predictions are more than predictions: Self-fulfilling performativity and the road towards morally responsible predictive systems. In Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency (FAccT ‘25), Athens, Greece. ACM. https://doi.org/10.1145/3715275.3732072

Pierson, E. (2015). Outnumbered but well-spoken: Female commenters in the New York Times. In Proceedings of the 18th ACM Conference on Computer Supported Cooperative Work & Social Computing (CSCW ‘15) (pp. 1201–1213). ACM. https://doi.org/10.1145/2675133.2675134

Friedkin, N. E., & Johnsen, E. C. (1990). Social influence and opinions. Journal of Mathematical Sociology, 15(3–4), 193–206. https://doi.org/10.1080/0022250X.1990.9990069
(This is the source for the stubborness term s * theta_i.)
