"""Average Copier and Nice Average Copier, ported from axelrod's averagecopier.py."""
import jax.numpy as jnp

from ipd.bots import Bot, _coop_with_prob
from ipd.game import C


def _average_copier_act(p0, n_coop, o, k):
    # p0 is the cooperation probability on the first move
    n_coop = n_coop + (o.their_last == C)
    p_coop = jnp.where(o.t == 0, p0, n_coop / jnp.maximum(o.t, 1))
    return _coop_with_prob(k, p_coop), n_coop


def _copier(name, p0):
    return Bot(name, lambda p, k: jnp.int32(0), _average_copier_act, jnp.float32(p0),
               memory_depth=float("inf"), stochastic=True)


average_copier = _copier("Average Copier", 0.5)
nice_average_copier = _copier("Nice Average Copier", 1.0)

STRATEGIES = [average_copier, nice_average_copier]
