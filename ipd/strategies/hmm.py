"""Hidden Markov model strategies, ported from axelrod's hmm.py.

One act serves every HMM: params hold the two transition matrices (conditioned on their last
move), the emission probabilities, the initial state and the opening move.
"""
import jax
import jax.numpy as jnp

from ipd.bots import Bot, _coop_with_prob
from ipd.game import C


def _hmm_act(p, state, o, k):
    k1, k2 = jax.random.split(k)
    row = jnp.where(o.their_last == C, p["t_C"][state], p["t_D"][state])
    nxt = jax.random.categorical(k1, jnp.log(row))
    action = jnp.where(o.t == 0, p["first"], _coop_with_prob(k2, p["emit"][nxt]))
    return action, jnp.where(o.t == 0, state, nxt).astype(jnp.int32)


def hmm(name, t_C, t_D, emissions, initial_state, initial_action, memory_depth=float("inf")):
    params = {"t_C": jnp.asarray(t_C, jnp.float32), "t_D": jnp.asarray(t_D, jnp.float32),
              "emit": jnp.asarray(emissions, jnp.float32), "init": jnp.int32(initial_state),
              "first": jnp.int32(initial_action)}
    return Bot(name, lambda p, k: p["init"], _hmm_act, params=params, memory_depth=memory_depth,
               stochastic=True)


evolved_hmm5 = hmm(
    "Evolved HMM 5",
    [[1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0.631, 0, 0, 0.369, 0],
     [0.143, 0.018, 0.118, 0, 0.721]],
    [[0, 1, 0, 0, 0],
     [0, 0.487, 0.513, 0, 0],
     [0, 0, 0, 0.590, 0.410],
     [1, 0, 0, 0, 0],
     [0, 0.287, 0.456, 0.146, 0.111]],
    [1, 0, 0, 1, 0.111],
    initial_state=3,
    initial_action=C,
    memory_depth=5,
)

STRATEGIES = [evolved_hmm5]
