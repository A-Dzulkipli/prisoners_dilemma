"""Meta strategies, ported from axelrod's meta.py.

A meta bot runs its whole team every turn and combines their proposals. As in the library, each team
member's own history is its own proposals (not the move the meta played), and MetaWinner scores each
member on (its proposal, the opponent's move). Teams are built from the library's class lists and
looked up in the registry, so a full team unrolls ~220 bots into one act.
"""
from functools import cache

import jax
import jax.numpy as jnp

from ipd.bots import Bot
from ipd.game import C, D

INF = float("inf")


@cache
def _library_team(kind):
    """Library team for a meta class: obey_axelrod strategies minus metas, filtered like meta.py."""
    import axelrod as axl
    from axelrod.classifier import Classifiers
    from axelrod.strategies.meta import MetaPlayer

    keep = {
        "all": lambda s: True,
        "memory_one": lambda s: Classifiers["memory_depth"](s()) <= 1,
        "finite_memory": lambda s: Classifiers["memory_depth"](s()) < INF,
        "long_memory": lambda s: Classifiers["memory_depth"](s()) == INF,
        "deterministic": lambda s: not Classifiers["stochastic"](s()),
        "stochastic": lambda s: Classifiers["stochastic"](s()),
    }[kind]
    ordinary = [s for s in axl.all_strategies if not issubclass(s, MetaPlayer) and Classifiers.obey_axelrod(s())]
    return tuple(s.name for s in ordinary if keep(s))


@cache
def _registry():
    from ipd.strategies import by_name

    return by_name()


def team(kind="all"):
    return [_registry()[name] for name in _library_team(kind)]


def _hunters(aggressive=False):
    names = ["Defector Hunter", "Alternator Hunter", "Random Hunter", "Math Constant Hunter", "Cycle Hunter",
             "Eventual Cycle Hunter"] + ["Cooperator Hunter"] * aggressive
    return [_registry()[n] for n in names]


def _meta_bot(name, members, choose, stochastic, nice=False):
    """choose(proposals, scores, obs, key) -> action. nice: never defect before the opponent has."""
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

    return Bot(name, init, act, memory_depth=INF, stochastic=stochastic)


def _majority(props, scores, o, k):
    n_d = props.sum()
    return jnp.where(2 * n_d > props.shape[0], D, C)


def _minority(props, scores, o, k):
    n_d = props.sum()
    return jnp.where(2 * n_d < props.shape[0], D, C)


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
    tft = jnp.where((o.t > 100) & (o.their_last == D), D, C)
    return jnp.where(jnp.any(props == D), D, tft)


def meta_majority(members=None, name="Meta Majority"):
    """Majority vote of the team, ties go to C."""
    members = team() if members is None else members
    return _meta_bot(name, members, _majority, any(b.stochastic for b in members))


def meta_minority(members=None, name="Meta Minority"):
    """Minority vote of the team, ties go to C."""
    members = team() if members is None else members
    return _meta_bot(name, members, _minority, any(b.stochastic for b in members))


def meta_winner(members=None, name="Meta Winner", nice=False):
    """Plays the proposal of the best scoring team member so far (C if any tied best proposes C)."""
    members = team() if members is None else members
    return _meta_bot(name, members, _winner, any(b.stochastic for b in members), nice)


def meta_winner_ensemble(members=None, name="Meta Winner Ensemble", nice=False):
    """Plays the proposal of a random member among the top 8% scorers so far."""
    members = team() if members is None else members
    return _meta_bot(name, members, _ensemble, True, nice)


def meta_mixer(members=None, name="Meta Mixer"):
    """Plays the proposal of a uniformly random team member each turn."""
    members = team() if members is None else members
    return _meta_bot(name, members, _mixer, True)


def meta_hunter(aggressive=False):
    """Defects if any hunter does; otherwise C, or TFT after 100 turns."""
    name = "Meta Hunter Aggressive" if aggressive else "Meta Hunter"
    return _meta_bot(name, _hunters(aggressive), _hunt, False)


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
    tft = jnp.where(o.t == 0, C, o.their_last)
    action = jnp.where(late, jnp.where(ncs < 0, D, C), tft)
    return action, (n_c.astype(jnp.int32), n_d.astype(jnp.int32))


def memory_decay(p_delete=0.1, p_alter=0.03, loss=-2.0, gain=1.0, duration=15):
    """TFT for 15 turns, then defects if the opponent's net cooperation score in a decaying memory is negative."""
    params = {"p_delete": jnp.float32(p_delete), "p_alter": jnp.float32(p_alter), "loss": jnp.float32(loss),
              "gain": jnp.float32(gain), "duration": jnp.int32(duration)}
    return Bot("Memory Decay", lambda p, k: (jnp.int32(0), jnp.int32(0)), _memory_decay_act, params=params,
               memory_depth=INF, stochastic=True)


# by_name() imports this module, so STRATEGIES must exist (empty) while the teams are being looked up
STRATEGIES = []
STRATEGIES += [
    memory_decay(),
    meta_hunter(),
    meta_hunter(aggressive=True),
    meta_majority(),
    meta_majority(team("memory_one"), "Meta Majority Memory One"),
    meta_majority(team("finite_memory"), "Meta Majority Finite Memory"),
    meta_majority(team("long_memory"), "Meta Majority Long Memory"),
    meta_minority(),
    meta_mixer(),
    meta_winner(),
    meta_winner(team("deterministic"), "Meta Winner Deterministic"),
    meta_winner_ensemble(),
    meta_winner(team("memory_one"), "Meta Winner Memory One"),
    meta_winner(team("finite_memory"), "Meta Winner Finite Memory"),
    meta_winner(team("long_memory"), "Meta Winner Long Memory"),
    meta_winner(team("stochastic"), "Meta Winner Stochastic"),
    meta_winner_ensemble(team("deterministic"), "NMWE Deterministic", nice=True),
    meta_winner_ensemble(team("finite_memory"), "NMWE Finite Memory", nice=True),
    meta_winner_ensemble(team("long_memory"), "NMWE Long Memory", nice=True),
    meta_winner_ensemble(team("memory_one"), "NMWE Memory One", nice=True),
    meta_winner_ensemble(team("stochastic"), "NMWE Stochastic", nice=True),
    meta_winner(name="Nice Meta Winner", nice=True),
    meta_winner_ensemble(name="Nice Meta Winner Ensemble", nice=True),
]
