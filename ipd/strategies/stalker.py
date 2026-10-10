"""Stalker, ported from axelrod's stalker.py."""
import jax.numpy as jnp

from ipd.bots import Bot, _coop_with_prob
from ipd.game import C, D


def _stalker_act(p, score, o, k):
    score = score + jnp.where(o.t > 0, o.my_payoff[jnp.maximum(o.my_last, 0), jnp.maximum(o.their_last, 0)], 0.0)
    (R, S), (T, P) = o.my_payoff
    n = jnp.maximum(o.t, 1)
    # average score compared with R, (R + P) / 2, 2 and 1, cleared of the division by n
    action = jnp.select(
        [o.t == 0, score > R * n, (2 * score > (R + P) * n) & (score < R * n), score > 2 * n, (score < 2 * n) & (score > n)],
        [C, D, C, C, D],
        _coop_with_prob(k, 0.5),
    )
    # FinalTransformer((D,)): defect on the last turn when the length is known
    return jnp.where((o.turns > 0) & (o.t >= o.turns - 1), D, action), score


stalker = Bot("Stalker", lambda p, k: jnp.float32(0), _stalker_act, memory_depth=float("inf"), stochastic=True)

STRATEGIES = [stalker]
