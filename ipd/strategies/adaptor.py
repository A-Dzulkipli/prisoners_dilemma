"""AdaptorBrief and AdaptorLong (Hauert 2002)."""
import jax.numpy as jnp

from ipd.bots import INF, Bot, coop_with_prob


def _adaptor_act(p, s, o, k):
    # p["delta"] is indexed by outcome (my move, their move); p_coop = 1 - perr inside [-1, 1), else perr
    s = s + jnp.where(o.t > 0, p["delta"][jnp.maximum(2 * o.my_last + o.their_last, 0)], 0.0)
    inside = (s >= -1) & (s < 1)
    return coop_with_prob(k, jnp.where(inside, 1 - p["perr"], p["perr"])), s


def adaptor(delta, perr=0.01, name="AbstractAdaptor"):
    """delta: additive state update after (CC, CD, DC, DD)."""
    params = {"delta": jnp.asarray(delta, jnp.float32), "perr": jnp.float32(perr)}
    return Bot(name, lambda p, k: jnp.float32(0), _adaptor_act, params, memory_depth=INF, stochastic=True)


adaptor_brief = adaptor([0.0, -1.001505, 0.992107, -0.638734], name="AdaptorBrief")
adaptor_long = adaptor([0.0, 1.888159, 1.858883, -0.995703], name="AdaptorLong")

STRATEGIES = [adaptor_brief, adaptor_long]
