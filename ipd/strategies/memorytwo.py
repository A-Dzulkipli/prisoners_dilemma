"""Memory-two strategies."""
import jax.numpy as jnp

from ipd.bots import INF, Bot, coop_with_prob, tft
from ipd.game import C, D


def _memory_two_init(p, k):
    # my move and their move two rounds ago
    return jnp.int32(C), jnp.int32(C)


def _memory_two_act(p, s, o, k):
    my2, their2 = s
    # index into the sixteen vector: (my t-2, my t-1, their t-2, their t-1) in binary, C = 0
    idx = 8 * my2 + 4 * jnp.maximum(o.my_last, 0) + 2 * their2 + jnp.maximum(o.their_last, 0)
    action = jnp.where(o.t < 2, p["initial"][jnp.minimum(o.t, 1)], coop_with_prob(k, p["p"][idx]))
    return action, (jnp.maximum(o.my_last, 0), jnp.maximum(o.their_last, 0))


def memory_two(sixteen_vector, initial=(C, C), name="Generic Memory Two Player"):
    """sixteen_vector: P(C | my last two, their last two), ordered as in axelrod's MemoryTwoPlayer."""
    params = {"p": jnp.asarray(sixteen_vector, jnp.float32), "initial": jnp.asarray(initial, jnp.int32)}
    return Bot(name, _memory_two_init, _memory_two_act, params=params, memory_depth=2)


aon2 = memory_two([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], name="AON2")
delayed_aon1 = memory_two([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], name="Delayed AON1")

TFT, TFTT, ALLD = 0, 1, 2


def _mem2_init(p, k):
    # mode, rounds until the next switch, times ALLD was chosen, my and their move two rounds ago
    return jnp.array([TFT, 3, 0, C, C], jnp.int32)


def _mem2_act(p, s, o, k):
    mode, shift, n_alld, my2, their2 = s
    shift = shift - 1
    switch = (shift == 0) & (n_alld < 2)
    # the last two rounds: both CC -> TFT, one CD and one DC -> TFTT, otherwise ALLD
    both_cc = (my2 == C) & (their2 == C) & (o.my_last == C) & (o.their_last == C)
    cd_dc = (my2 != their2) & (o.my_last != o.their_last) & (my2 != o.my_last)
    new_mode = jnp.select([both_cc, cd_dc], [TFT, TFTT], ALLD)
    mode = jnp.where(switch, new_mode, mode)
    n_alld = n_alld + (switch & (new_mode == ALLD))
    shift = jnp.where(switch, 2, shift)
    action = jnp.select(
        [mode == TFT, mode == TFTT],
        [tft(o), jnp.where((o.their_last == D) & (their2 == D), D, C)],
        D,
    )
    new_s = jnp.stack([mode, shift, n_alld, jnp.maximum(o.my_last, 0), jnp.maximum(o.their_last, 0)])
    return action, new_s


mem2 = Bot("MEM2", _mem2_init, _mem2_act, memory_depth=INF)

STRATEGIES = [aon2, delayed_aon1, mem2]
