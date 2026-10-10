"""Better and Better, ported from axelrod's better_and_better.py."""
from ipd.bots import Bot, _coop_with_prob, no_state

# cooperates with probability (t + 1) / 1000
better_and_better = Bot("Better and Better", no_state, lambda p, s, o, k: (_coop_with_prob(k, (o.t + 1) / 1000), s),
                        memory_depth=float("inf"), stochastic=True)

STRATEGIES = [better_and_better]
