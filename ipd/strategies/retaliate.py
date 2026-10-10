"""Retaliate variants."""
import jax.numpy as jnp

from ipd.bots import INF, Bot
from ipd.game import C, D


def _count_wins(n_cd, n_dc, o):
    # n_cd: rounds of my C vs their D; n_dc: the reverse
    n_cd = n_cd + ((o.my_last == C) & (o.their_last == D))
    n_dc = n_dc + ((o.my_last == D) & (o.their_last == C))
    return n_cd, n_dc


def _retaliate_act(p, s, o, k):
    n_cd, n_dc = _count_wins(s[0], s[1], o)
    # CD > DC * threshold, with the threshold in percent so it stays exact
    return jnp.where(100 * n_cd > n_dc * p, D, C), jnp.stack([n_cd, n_dc])


def retaliate(name="Retaliate", threshold_pct=10):
    """Defect while the opponent's D-vs-my-C wins exceed threshold_pct percent of mine."""
    return Bot(name, lambda p, k: jnp.zeros(2, jnp.int32), _retaliate_act, jnp.int32(threshold_pct), memory_depth=INF)


def _limited_retaliate_act(p, s, o, k):
    n_cd, n_dc, count = s
    n_cd, n_dc = _count_wins(n_cd, n_dc, o)
    retaliating = 100 * n_cd > n_dc * p["pct"]
    hit = retaliating & (count < p["limit"])
    # at the limit the count resets and the next round may start retaliating again
    count = jnp.where(hit, count + 1, 0)
    return jnp.where(hit, D, C), jnp.stack([n_cd, n_dc, count])


def limited_retaliate(name="Limited Retaliate", threshold_pct=10, limit=20):
    """Retaliate, but cooperate again after limit consecutive retaliations."""
    params = {"pct": jnp.int32(threshold_pct), "limit": jnp.int32(limit)}
    return Bot(name, lambda p, k: jnp.zeros(3, jnp.int32), _limited_retaliate_act, params, memory_depth=INF)


STRATEGIES = [
    limited_retaliate(),
    limited_retaliate("Limited Retaliate 2", 8, 15),
    limited_retaliate("Limited Retaliate 3", 5, 20),
    retaliate(),
    retaliate("Retaliate 2", 8),
    retaliate("Retaliate 3", 5),
]
