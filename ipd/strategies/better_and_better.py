"""Better and Better."""
from ipd.bots import INF, Bot, coop_with_prob, no_state

# cooperates with probability (t + 1) / 1000
better_and_better = Bot("Better and Better", no_state, lambda p, s, o, k: (coop_with_prob(k, (o.t + 1) / 1000), s),
                        memory_depth=INF, stochastic=True)

STRATEGIES = [better_and_better]
