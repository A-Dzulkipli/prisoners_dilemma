"""Stalker."""
import jax.numpy as jnp

from ipd.bots import INF, Bot, coop_with_prob, defect_at_end, last_payoff
from ipd.game import C, D


def _stalker_act(p, score, o, k):
    score = score + last_payoff(o)
    (R, S), (T, P) = o.my_payoff
    n = jnp.maximum(o.t, 1)
    # average score compared with R, (R + P) / 2, 2 and 1, cleared of the division by n
    action = jnp.select(
        [o.t == 0, score > R * n, (2 * score > (R + P) * n) & (score < R * n), score > 2 * n,
         (score < 2 * n) & (score > n)],
        [C, D, C, C, D],
        coop_with_prob(k, 0.5),
    )
    return defect_at_end(o, action), score


stalker = Bot("Stalker", lambda p, k: jnp.float32(0), _stalker_act, memory_depth=INF, stochastic=True)

STRATEGIES = [stalker]
