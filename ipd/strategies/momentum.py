"""Momentum, ported from axelrod's momentum.py."""
import jax.numpy as jnp
import numpy as np

from ipd.bots import Bot
from ipd.game import C, D
from ipd.strategies.dbs import _df_add, _df_mul


def _df(x):
    """A float64 constant as a (hi, lo) float32 pair."""
    hi = np.float32(x)
    return jnp.float32(hi), jnp.float32(x - float(hi))


def _momentum_act(p, m, o, k):
    # momentum is a double-float: in float32 its gap to the threshold can drown in rounding
    coop = (o.their_last == C).astype(jnp.float32)
    step = _df_add(_df_mul(p["alpha"], m), (coop * p["one_minus_alpha"][0], coop * p["one_minus_alpha"][1]))
    m = tuple(jnp.where(o.t > 0, new, old) for new, old in zip(step, m))
    gap = _df_add(m, (-p["threshold"][0], -p["threshold"][1]))
    return jnp.where((o.t == 0) | (gap[0] + gap[1] >= 0), C, D), m


def momentum(alpha=0.9914655399877477, threshold=0.9676595613724907):
    params = {"alpha": _df(alpha), "one_minus_alpha": _df(1 - alpha), "threshold": _df(threshold)}
    return Bot("Momentum", lambda p, k: (jnp.float32(1), jnp.float32(0)), _momentum_act, params,
               memory_depth=float("inf"))


STRATEGIES = [momentum()]
