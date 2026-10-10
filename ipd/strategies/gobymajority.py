"""Go By Majority family, ported from axelrod's gobymajority.py."""
import jax.numpy as jnp

from ipd.bots import Bot
from ipd.game import C, D

INF = float("inf")


def _gbm_init(p, k):
    # opponent's last `window` moves (-1 = empty, unused when unbounded) and defections minus cooperations
    return jnp.full(p["window"].shape, -1, jnp.int32), jnp.int32(0)


def _gbm_act(p, s, o, k):
    buf, d_minus_c = s
    d_minus_c = d_minus_c + (o.their_last == D) - (o.their_last == C)
    window = buf.shape[0]
    if window:  # static: the buffer length is the memory
        slot = (o.t - 1) % window
        leaving = jnp.where(o.t > 0, buf[slot], -1)
        d_minus_c = d_minus_c - (leaving == D) + (leaving == C)
        buf = jnp.where(o.t > 0, buf.at[slot].set(o.their_last), buf)
    action = jnp.where((d_minus_c > 0) | ((d_minus_c == 0) & ~p["soft"]), D, C)
    return action, (buf, d_minus_c.astype(jnp.int32))


def go_by_majority(memory_depth=INF, soft=True, name="Go By Majority"):
    """Defect when the opponent defected more than cooperated (in the last memory_depth rounds); soft breaks ties with C."""
    window = 0 if memory_depth == INF else memory_depth
    params = {"window": jnp.zeros(window, jnp.int32), "soft": jnp.bool_(soft)}
    return Bot(name, _gbm_init, _gbm_act, params=params, memory_depth=memory_depth)


STRATEGIES = [
    go_by_majority(),
    go_by_majority(10, name="Go By Majority 10"),
    go_by_majority(20, name="Go By Majority 20"),
    go_by_majority(40, name="Go By Majority 40"),
    go_by_majority(5, name="Go By Majority 5"),
    go_by_majority(soft=False, name="Hard Go By Majority"),
    go_by_majority(10, False, "Hard Go By Majority 10"),
    go_by_majority(20, False, "Hard Go By Majority 20"),
    go_by_majority(40, False, "Hard Go By Majority 40"),
    go_by_majority(5, False, "Hard Go By Majority 5"),
]
