"""Prober strategies."""
import jax.numpy as jnp

from ipd.bots import INF, Bot, coop_with_prob, no_state, tft
from ipd.game import C, D


def _first_moves_init(p, k):
    return jnp.full(4, -1, jnp.int32)


def _record_first(first, o):
    """Keep the opponent's first four moves."""
    t = o.t - 1
    return jnp.where((o.t > 0) & (t < 4), first.at[jnp.clip(t, 0, 3)].set(o.their_last), first)


def _collective_act(p, s, o, k):
    first, n_def = s
    first = _record_first(first, o)
    n_def = n_def + (o.their_last == D)
    late = jnp.where((n_def <= 1) & (first[0] == C) & (first[1] == D), C, D)
    return jnp.select([o.t == 0, o.t == 1], [C, D], late), (first, n_def)


collective_strategy = Bot(
    "CollectiveStrategy", lambda p, k: (_first_moves_init(p, k), jnp.int32(0)), _collective_act, memory_depth=INF
)

DETECTIVE_OPENING = jnp.array([C, D, C, C], jnp.int32)


def _detective_act(p, first, o, k):
    first = _record_first(first, o)
    late = jnp.where(jnp.any(first == D), tft(o), D)
    return jnp.where(o.t < 4, DETECTIVE_OPENING[jnp.minimum(o.t, 3)], late), first


detective = Bot("Detective", _first_moves_init, _detective_act, memory_depth=INF)


def _opening_act(p, first, o, k):
    """Play `opening`, then defect forever if the opponent's moves 2-3 (1-indexed) match `trigger`, else TFT.

    trigger entries are C/D, or -1 for 'any'. With Prober 2 the trigger instead means cooperate forever.
    """
    first = _record_first(first, o)
    n = p["opening"].shape[0]
    hit = jnp.all((p["trigger"] == -1) | (p["trigger"] == first[1:3]))
    late = jnp.where(hit, p["on_hit"], tft(o))
    return jnp.where(o.t < n, p["opening"][jnp.minimum(o.t, n - 1)], late), first


def _opening_bot(name, opening, trigger, on_hit=D):
    params = {"opening": jnp.asarray(opening, jnp.int32), "trigger": jnp.asarray(trigger, jnp.int32),
              "on_hit": jnp.int32(on_hit)}
    return Bot(name, _first_moves_init, _opening_act, params=params, memory_depth=INF)


prober = _opening_bot("Prober", [D, C, C], [C, C])
prober2 = _opening_bot("Prober 2", [D, C, C], [D, C], on_hit=C)
prober3 = _opening_bot("Prober 3", [D, C], [C, -1])
hard_prober = _opening_bot("Hard Prober", [D, D, C, C], [C, C])

PROBER4_OPENING = jnp.array([C, C, D, C, D, D, D, C, C, D, C, D, C, C, D, C, D, D, C, D], jnp.int32)


def _prober4_act(p, s, o, k):
    just, unjust, defector = s
    # the library counts the opponent's defections in rounds 0..18 only (it stops updating at turn 20)
    counting = (o.t >= 1) & (o.t < 20) & (o.their_last == D)
    just = just + (counting & (o.my_last == D))
    unjust = unjust + (counting & (o.my_last == C))
    defector = jnp.where(o.t == 20, jnp.abs(just - unjust) <= 2, defector)
    action = jnp.select(
        [o.t < 20, defector, o.t < 25], [PROBER4_OPENING[jnp.minimum(o.t, 19)], D, C], tft(o)
    )
    return action, (just, unjust, defector)


prober4 = Bot("Prober 4", lambda p, k: (jnp.int32(0), jnp.int32(0), jnp.bool_(False)), _prober4_act, memory_depth=INF)


def _naive_prober_act(p, s, o, k):
    action = jnp.where(o.t == 0, C, jnp.where(o.their_last == D, D, coop_with_prob(k, 1 - p)))
    return action, s


def naive_prober(p=0.1):
    """TFT that also defects with probability p after the opponent cooperates."""
    return Bot("Naive Prober", no_state, _naive_prober_act, params=jnp.float32(p), stochastic=True)


def _remorseful_act(p, probing, o, k):
    random_move = coop_with_prob(k, 1 - p)
    # after a probing defection, forgive one retaliation
    action = jnp.select([o.t == 0, (o.their_last == D) & probing, o.their_last == D], [C, C, D], random_move)
    probing = jnp.where((o.t > 0) & (o.their_last != D), random_move == D, jnp.where(o.their_last == D, False, probing))
    return action, probing


def remorseful_prober(p=0.1):
    """Naive Prober that does not retaliate against a reply to its own random defection."""
    return Bot(
        "Remorseful Prober", lambda p, k: jnp.bool_(False), _remorseful_act, params=jnp.float32(p),
        memory_depth=2, stochastic=True,
    )


STRATEGIES = [
    collective_strategy,
    detective,
    hard_prober,
    naive_prober(),
    prober,
    prober2,
    prober3,
    prober4,
    remorseful_prober(),
]
