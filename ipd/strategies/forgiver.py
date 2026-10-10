"""Forgiver and Forgiving Tit For Tat, ported from axelrod's strategies/forgiver.py."""
import jax.numpy as jnp

from ipd.bots import Bot
from ipd.game import C, D

INF = float("inf")


def _defects_over_10pct(n_def, o):
    n_def = n_def + (o.their_last == D)
    return 10 * n_def > o.t, n_def.astype(jnp.int32)


def _forgiver_act(p, n_def, o, k):
    bad, n_def = _defects_over_10pct(n_def, o)
    return jnp.where(bad, D, C), n_def


def _forgiving_tft_act(p, n_def, o, k):
    bad, n_def = _defects_over_10pct(n_def, o)
    return jnp.where(bad, o.their_last, C), n_def


forgiver = Bot("Forgiver", lambda p, k: jnp.int32(0), _forgiver_act, memory_depth=INF)
forgiving_tit_for_tat = Bot("Forgiving Tit For Tat", lambda p, k: jnp.int32(0), _forgiving_tft_act, memory_depth=INF)

STRATEGIES = [forgiver, forgiving_tit_for_tat]
