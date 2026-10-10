"""Finite state machine strategies, ported from axelrod's finite_state_machines.py.

One act serves every machine: params hold dense next-state and output tables indexed by
[state, their last move], so machines with the same number of states can be vmapped over params.
"""
import jax.numpy as jnp
import numpy as np

from ipd.bots import Bot


def _fsm_act(p, state, o, k):
    nxt = p["next"][state, jnp.maximum(o.their_last, 0)]
    out = p["out"][state, jnp.maximum(o.their_last, 0)]
    first = o.t == 0
    return jnp.where(first, p["first"], out), jnp.where(first, state, nxt).astype(jnp.int32)


def fsm(name, transitions, initial_state, initial_action, n_states=None, memory_depth=float("inf")):
    """transitions: rows (state, their last move, next state, my move), as in axelrod. Pad to n_states."""
    labels = sorted({row[0] for row in transitions})
    idx = {s: i for i, s in enumerate(labels)}
    n = n_states or len(labels)
    nxt, out = np.zeros((n, 2), np.int32), np.zeros((n, 2), np.int32)
    for s, their, s2, mine in transitions:
        nxt[idx[s], their] = idx[s2]
        out[idx[s], their] = mine
    params = {"next": jnp.asarray(nxt), "out": jnp.asarray(out), "first": jnp.int32(initial_action),
              "init": jnp.int32(idx[initial_state])}
    return Bot(name, lambda p, k: p["init"], _fsm_act, params=params, memory_depth=memory_depth)


def from_axelrod(cls_name):
    """Reads the transition table of an axelrod FSMPlayer subclass (lazy import)."""
    import axelrod as axl

    cls = getattr(axl, cls_name)
    player = cls()
    rows = [(s, int(a == axl.Action.D), s2, int(b == axl.Action.D)) for (s, a), (s2, b) in
            player.fsm.state_transitions.items()]
    return fsm(cls.name, rows, player.initial_state, int(player.initial_action == axl.Action.D),
               memory_depth=cls.classifier["memory_depth"])


NAMES = [
    "EvolvedFSM16", "EvolvedFSM16Noise05", "EvolvedFSM4", "EvolvedFSM6", "Fortress3", "Fortress4",
    "Predator", "Pun1", "Raider", "Ripoff", "SolutionB1", "SolutionB5", "TF1", "TF2", "TF3",
    "Thumper", "UsuallyCooperates", "UsuallyDefects",
]
STRATEGIES = [from_axelrod(n) for n in NAMES]
