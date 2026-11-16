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

# Model outline

We build on the opinion dynamics model of Baumann et al. (2020). Every agent has an opinion `x_i` on one axis and an activity `a_i` (how likely they are to post in a given time step). Opinions change as

```
dx_i/dt = -x_i + s * theta_i + K * sum_j A_ij(t) * tanh(alpha * x_j)
```

- `-x_i` is forgetting: without input, opinions decay toward 0.
- `s * theta_i` (our addition) It pulls each agent back toward their original opinion `theta_i` (Friedkin & Johnsen 1990). With `s = 0` this is plain Baumann. With `s = 1`, an agent left alone returns to what they originally thought.
- The last term is social influence from whoever reached agent `i` in this time step. `K` sets its strength, and `alpha` sets how controversial the topic is (how strongly even moderate opinions push).

Active agents contact `m = 10` others, preferring people with similar opinions (homophily `beta`), and with probability `r` the contact also influences them back.

Our difference from Baumann: matching algorithm of who interacts with who. In Baumann's model, partners are picked by comparing true opinions. In ours, the platform picks them, and it only has estimates `x_hat` built from noisy observations of agents when they are active. Each estimate mixes the agent's own history with the average over all observations:

```
x_hat_i = w_i * own_mean_i + (1 - w_i) * pooled_mean,    w_i = n_i / (n_i + lam)
```

## References

Baumann, F., Lorenz-Spreen, P., Sokolov, I. M., & Starnini, M. (2020). Modeling echo chambers and polarization dynamics in social networks. Physical Review Letters, 124(4), 048301. https://doi.org/10.1103/PhysRevLett.124.048301

Khosrowi, D., Ahlers, M., & van Basshuysen, P. (2025). When predictions are more than predictions: Self-fulfilling performativity and the road towards morally responsible predictive systems. In Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency (FAccT ‘25), Athens, Greece. ACM. https://doi.org/10.1145/3715275.3732072

Pierson, E. (2015). Outnumbered but well-spoken: Female commenters in the New York Times. In Proceedings of the 18th ACM Conference on Computer Supported Cooperative Work & Social Computing (CSCW ‘15) (pp. 1201–1213). ACM. https://doi.org/10.1145/2675133.2675134

Friedkin, N. E., & Johnsen, E. C. (1990). Social influence and opinions. Journal of Mathematical Sociology, 15(3–4), 193–206. https://doi.org/10.1080/0022250X.1990.9990069
(This is the source for the stubborness term s * theta_i.)
