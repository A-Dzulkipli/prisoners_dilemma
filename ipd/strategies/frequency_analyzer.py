"""FrequencyAnalyzer, ported from axelrod's frequency_analyzer.py."""
import jax.numpy as jnp

from ipd.bots import Bot
from ipd.game import C, D


def _init(p, k):
    # per 4-move key: times seen, opponent cooperations that followed; the last two rounds as codes
    return jnp.zeros(16, jnp.int32), jnp.zeros(16, jnp.int32), jnp.int32(0), jnp.int32(0)


def _frequency_analyzer_act(p, s, o, k):
    seen, coops, round2, round1 = s  # round codes are 2 * their + mine for rounds t-2 and t-1
    round0 = 2 * jnp.maximum(o.their_last, 0) + jnp.maximum(o.my_last, 0)
    last_key, cur_key = 4 * round2 + round1, 4 * round1 + round0
    update = o.t > 5
    seen = seen.at[last_key].add(update)
    coops = coops.at[last_key].add(update & (o.their_last == C))
    tft = jnp.where(o.their_last == D, D, C)
    # the library divides by the match length so far, not by the times the key was seen
    use_table = (o.t >= 30) & (seen[cur_key] > 0)
    action = jnp.where(use_table, jnp.where(4 * coops[cur_key] > o.t, C, D), jnp.where(o.t == 0, C, tft))
    return action, (seen, coops, round1, round0)


frequency_analyzer = Bot("FrequencyAnalyzer", _init, _frequency_analyzer_act, memory_depth=float("inf"))

STRATEGIES = [frequency_analyzer]
