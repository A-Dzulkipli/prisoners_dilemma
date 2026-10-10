"""Meta strategies.

A meta bot runs its whole team every turn and combines their proposals. As in the library, each team
member's own history is its own proposals (not the move the meta played), and MetaWinner scores each
member on (its proposal, the opponent's move). Teams are built from the library's class lists and
looked up in the registry, so a full team unrolls ~220 bots into one act.
"""
from functools import cache

import jax
import jax.numpy as jnp

from ipd.bots import INF, Bot, tft
from ipd.game import C, D

HUNTERS = ("Defector Hunter", "Alternator Hunter", "Random Hunter", "Math Constant Hunter", "Cycle Hunter",
           "Eventual Cycle Hunter")


@cache
def _library_team(kind):
    """Library team for a meta class: obey_axelrod strategies minus metas, filtered like meta.py."""
    import axelrod as axl
    from axelrod.strategies.meta import MetaPlayer

    depth, stochastic = axl.Classifiers["memory_depth"], axl.Classifiers["stochastic"]
    keep = {
        "all": lambda s: True,
        "memory_one": lambda s: depth(s) <= 1,
        "finite_memory": lambda s: depth(s) < INF,
        "long_memory": lambda s: depth(s) == INF,
        "deterministic": lambda s: not stochastic(s),
        "stochastic": stochastic,
    }[kind]
    # class names: some instances rename themselves (e.g. "Soft Go By Majority")
    players = [(s.name, s()) for s in axl.all_strategies if not issubclass(s, MetaPlayer)]
    return tuple(name for name, s in players if axl.Classifiers.obey_axelrod(s) and keep(s))


@cache
def _registry():
    from ipd.strategies import by_name

    return by_name()


def team(kind="all"):
    return [_registry()[name] for name in _library_team(kind)]


def meta_bot(name, members, choose, nice=False):
    """Run every member each turn and play choose(proposals, scores, obs, key).

    nice: never defect before the opponent has.
    """
    n = len(members)

    def init(p, key):
        keys = jax.random.split(key, n)
        states = tuple(b.init(b.params, k) for b, k in zip(members, keys))
        # member states, last proposals (each member's own history), running scores, opponent has defected
        return states, jnp.full(n, -1, jnp.int32), jnp.zeros(n, jnp.float32), jnp.bool_(False)

    def act(p, s, o, key):
        states, last, scores, wronged = s
        scores = scores + jnp.where(o.t > 0, o.my_payoff[jnp.maximum(last, 0), jnp.maximum(o.their_last, 0)], 0.0)
        wronged = wronged | (o.their_last == D)
        k_meta, k_team = jax.random.split(key)
        new_states, proposals = [], []
        for b, st, mine, k in zip(members, states, last, jax.random.split(k_team, n)):
            a, st = b.act(b.params, st, o._replace(my_last=mine), k)
            new_states.append(st)
            proposals.append(jnp.asarray(a, jnp.int32))
        proposals = jnp.stack(proposals)
        action = choose(proposals, scores, o, k_meta)
        if nice:
            action = jnp.where(wronged, action, C)
        return action, (tuple(new_states), proposals, scores, wronged)

    stochastic = choose in (_ensemble, _mixer) or any(b.stochastic for b in members)
    return Bot(name, init, act, memory_depth=INF, stochastic=stochastic)


def _majority(props, scores, o, k):
    # ties go to C
    return jnp.where(2 * props.sum() > props.shape[0], D, C)


def _minority(props, scores, o, k):
    return jnp.where(2 * props.sum() < props.shape[0], D, C)


def _winner(props, scores, o, k):
    # C if any of the top scorers proposes C
    best = scores == scores.max()
    return jnp.where(jnp.any(best & (props == C)), C, D)


def _ensemble(props, scores, o, k):
    # library sorts (score, index) descending and picks uniformly among the top 8%
    n = props.shape[0]
    top = max(1, int(n * 0.08))
    order = jnp.lexsort((jnp.arange(n), scores))[::-1]
    return props[order[jax.random.randint(k, (), 0, top)]]


def _mixer(props, scores, o, k):
    return props[jax.random.randint(k, (), 0, props.shape[0])]


def _hunt(props, scores, o, k):
    # D if any hunter proposes D, else C, or TFT after 100 turns
    tft_late = jnp.where((o.t > 100) & (o.their_last == D), D, C)
    return jnp.where(jnp.any(props == D), D, tft_late)


def _memory_decay_act(p, s, o, k):
    n_c, n_d = s
    # memory is a multiset of the opponent's moves; only the counts matter
    n_c = n_c + (o.their_last == C)
    n_d = n_d + (o.their_last == D)
    k1, k2, k3, k4 = jax.random.split(k, 4)
    late = o.t >= p["duration"]
    # alter then delete a uniformly random entry
    alter = late & (jax.random.uniform(k1) <= p["p_alter"])
    pick_c = jax.random.uniform(k2) * (n_c + n_d) < n_c
    n_c, n_d = n_c + alter * jnp.where(pick_c, -1, 1), n_d + alter * jnp.where(pick_c, 1, -1)
    delete = late & (jax.random.uniform(k3) <= p["p_delete"])
    pick_c = jax.random.uniform(k4) * (n_c + n_d) < n_c
    n_c, n_d = n_c - delete * pick_c, n_d - delete * ~pick_c
    ncs = p["gain"] * n_c + p["loss"] * n_d
    return jnp.where(late, jnp.where(ncs < 0, D, C), tft(o)), (n_c, n_d)


def memory_decay(p_delete=0.1, p_alter=0.03, loss=-2.0, gain=1.0, duration=15):
    """TFT for 15 turns, then defects if the opponent's net cooperation score in a decaying memory is negative."""
    params = {"p_delete": jnp.float32(p_delete), "p_alter": jnp.float32(p_alter), "loss": jnp.float32(loss),
              "gain": jnp.float32(gain), "duration": jnp.int32(duration)}
    return Bot("Memory Decay", lambda p, k: (jnp.int32(0), jnp.int32(0)), _memory_decay_act, params=params,
               memory_depth=INF, stochastic=True)


# by_name() imports this module, so STRATEGIES must exist (empty) while the teams are being looked up
STRATEGIES = []
_hunters = [_registry()[name] for name in HUNTERS]
STRATEGIES += [
    memory_decay(),
    meta_bot("Meta Hunter", _hunters, _hunt),
    meta_bot("Meta Hunter Aggressive", _hunters + [_registry()["Cooperator Hunter"]], _hunt),
    meta_bot("Meta Majority", team(), _majority),
    meta_bot("Meta Majority Memory One", team("memory_one"), _majority),
    meta_bot("Meta Majority Finite Memory", team("finite_memory"), _majority),
    meta_bot("Meta Majority Long Memory", team("long_memory"), _majority),
    meta_bot("Meta Minority", team(), _minority),
    meta_bot("Meta Mixer", team(), _mixer),
    meta_bot("Meta Winner", team(), _winner),
    meta_bot("Meta Winner Deterministic", team("deterministic"), _winner),
    meta_bot("Meta Winner Ensemble", team(), _ensemble),
    meta_bot("Meta Winner Memory One", team("memory_one"), _winner),
    meta_bot("Meta Winner Finite Memory", team("finite_memory"), _winner),
    meta_bot("Meta Winner Long Memory", team("long_memory"), _winner),
    meta_bot("Meta Winner Stochastic", team("stochastic"), _winner),
    meta_bot("NMWE Deterministic", team("deterministic"), _ensemble, nice=True),
    meta_bot("NMWE Finite Memory", team("finite_memory"), _ensemble, nice=True),
    meta_bot("NMWE Long Memory", team("long_memory"), _ensemble, nice=True),
    meta_bot("NMWE Memory One", team("memory_one"), _ensemble, nice=True),
    meta_bot("NMWE Stochastic", team("stochastic"), _ensemble, nice=True),
    meta_bot("Nice Meta Winner", team(), _winner, nice=True),
    meta_bot("Nice Meta Winner Ensemble", team(), _ensemble, nice=True),
]
