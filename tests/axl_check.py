"""Check JAX bots against their axelrod library counterparts.

Deterministic bots must match move for move: against scripted random opponents (via MockPlayer)
and against the classic reactive bots. Stochastic bots are compared on mean cooperation and score.
"""
import axelrod as axl
import jax
import jax.numpy as jnp
import numpy as np
from axelrod.mock_player import MockPlayer

from ipd import bots
from ipd.engine import play_match
from ipd.game import C, D, Config

KEY = jax.random.key(0)
AXL = {s.name: s for s in axl.all_strategies}
TO_AXL = {C: axl.Action.C, D: axl.Action.D}
CLASSIC_AXL = [
    (bots.cooperator, axl.Cooperator),
    (bots.defector, axl.Defector),
    (bots.alternator, axl.Alternator),
    (bots.tit_for_tat, axl.TitForTat),
    (bots.suspicious_tit_for_tat, axl.SuspiciousTitForTat),
    (bots.win_stay_lose_shift, axl.WinStayLoseShift),
    (bots.tit_for_two_tats, axl.TitFor2Tats),
    (bots.grudger, axl.Grudger),
]


def sequence_bot(actions):
    return bots.Bot("Sequence", bots.no_state, lambda p, s, o, k: (p[o.t], s), params=jnp.asarray(actions))


def opponent_sequences(n=40, turns=200, seed=0):
    """Random sequences with cooperation rates from 5% to 95%, half of them in streaky blocks."""
    rng = np.random.default_rng(seed)
    seqs = []
    for i, p in enumerate(np.linspace(0.05, 0.95, n)):
        if i % 2:
            blocks = rng.integers(1, 12, turns)
            moves = np.repeat(rng.random(turns) > p, blocks)[:turns]
        else:
            moves = rng.random(turns) > p
        seqs.append(moves.astype(np.int32))
    return np.stack(seqs)


def _to_np(history, idx=0):
    return np.array([int(pair[idx] == axl.Action.D) for pair in history])


def _report(ours, theirs, label):
    if not np.array_equal(ours, theirs):
        t = int(np.argmax(ours != theirs))
        lo = max(0, t - 5)
        raise AssertionError(
            f"{label}: first mismatch at turn {t}\n ours   {ours[lo:t + 3]}\n theirs {theirs[lo:t + 3]}"
        )


def assert_same_moves(bot, axl_cls, turns=200, n_seqs=40, seqs=None):
    """Exact move-for-move check for a deterministic bot, against seqs (default: random sequences) and the classics."""
    cfg = Config(turns=turns)
    seqs = opponent_sequences(n_seqs, turns) if seqs is None else np.asarray(seqs)
    ours = jax.jit(jax.vmap(lambda s: play_match(bot, sequence_bot(s), KEY, cfg).actions[:, 0]))(seqs)
    for i, seq in enumerate(seqs):
        history = axl.Match((axl_cls(), MockPlayer([TO_AXL[int(a)] for a in seq])), turns=turns).play()
        _report(np.asarray(ours[i]), _to_np(history), f"{bot.name} vs sequence {i}")
    for opp, opp_axl in CLASSIC_AXL + [(bot, axl_cls)]:
        ours = np.asarray(play_match(bot, opp, KEY, cfg).actions[:, 0])
        _report(ours, _to_np(axl.Match((axl_cls(), opp_axl()), turns=turns).play()), f"{bot.name} vs {opp.name}")


def assert_same_distribution(bot, axl_cls, turns=200, n=300, coop_tol=0.04, score_tol=0.12):
    """Statistical check for a stochastic bot: mean cooperation and score per opponent."""
    cfg = Config(turns=turns)
    keys = jax.random.split(KEY, n)
    for opp, opp_axl in CLASSIC_AXL + [(bot, axl_cls)]:
        r = jax.jit(jax.vmap(lambda k: play_match(bot, opp, k, cfg)))(keys)
        ours_coop = float((r.actions[:, :, 0] == C).mean())
        ours_score = float(r.scores[:, :, 0].mean())
        coop, score = [], []
        for seed in range(n):
            m = axl.Match((axl_cls(), opp_axl()), turns=turns, seed=seed)
            m.play()
            coop.append(m.normalised_cooperation()[0])
            score.append(m.final_score_per_turn()[0])
        label = f"{bot.name} vs {opp.name}"
        assert abs(ours_coop - np.mean(coop)) < coop_tol, f"{label}: coop {ours_coop:.3f} vs {np.mean(coop):.3f}"
        assert abs(ours_score - np.mean(score)) < score_tol, f"{label}: score {ours_score:.3f} vs {np.mean(score):.3f}"
