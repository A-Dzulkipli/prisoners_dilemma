"""Bush-Mosteller reinforcement learner, ported from axelrod's bush_mosteller.py."""
import jax.numpy as jnp

from ipd.bots import Bot, _coop_with_prob
from ipd.game import C


def _bm_act(p, s, o, k):
    c_init, d_init, divider, lr = p
    c_prob, d_prob = s
    best = o.my_payoff.max()  # max of R, P, S, T
    aspiration = jnp.abs(best / divider)
    payoff = o.my_payoff[jnp.maximum(o.my_last, 0), jnp.maximum(o.their_last, 0)]
    stim = jnp.maximum((payoff - aspiration) / jnp.abs(best - aspiration), -1.0)
    prob = jnp.where(o.my_last == C, c_prob, d_prob)
    prob = prob + lr * stim * jnp.where(stim >= 0, 1 - prob, prob)
    live = o.t > 0
    c_prob = jnp.where(live & (o.my_last == C), prob, c_prob)
    d_prob = jnp.where(live & (o.my_last != C), prob, d_prob)
    return _coop_with_prob(k, c_prob / (c_prob + d_prob)), jnp.stack([c_prob, d_prob])


def bush_mosteller(c_prob=0.5, d_prob=0.5, aspiration_level_divider=3.0, learning_rate=0.5):
    params = jnp.array([c_prob, d_prob, aspiration_level_divider, learning_rate], jnp.float32)
    return Bot("Bush Mosteller", lambda p, k: p[:2], _bm_act, params=params,
               memory_depth=float("inf"), stochastic=True)


STRATEGIES = [bush_mosteller()]
