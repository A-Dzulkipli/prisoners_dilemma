"""Neural network strategies, ported from axelrod's ann.py.

One act serves every network: a single hidden ReLU layer over 17 history features, with the
hidden layer zero-padded to a common width so the variants share one params shape.
"""
import jax.numpy as jnp
import numpy as np

from ipd.bots import Bot
from ipd.game import C, D


def _ann_init(p, k):
    # their first two moves, my and their move at t-2, their coops, their defs, my coops, my defs
    return jnp.zeros(8, jnp.int32)


def _one_hot(move, seen):
    return jnp.stack([seen & (move == C), seen & (move == D)])


def _ann_act(p, s, o, k):
    first0, first1, my_prev2, their_prev2, their_c, their_d, my_c, my_d = s
    their_c, their_d = their_c + (o.their_last == C), their_d + (o.their_last == D)
    my_c, my_d = my_c + (o.my_last == C), my_d + (o.my_last == D)
    first0 = jnp.where(o.t == 1, o.their_last, first0)
    first1 = jnp.where(o.t == 2, o.their_last, first1)
    x = jnp.concatenate([
        _one_hot(first0, o.t >= 1), _one_hot(first1, o.t >= 2),
        _one_hot(o.my_last, o.t >= 1), _one_hot(my_prev2, o.t >= 2),
        _one_hot(o.their_last, o.t >= 1), _one_hot(their_prev2, o.t >= 2),
        jnp.stack([their_c, their_d, my_c, my_d, o.t]),
    ]).astype(jnp.float32)
    pre = p["w_in"] @ x + p["bias"]
    # the library's relu is np.vectorize(lambda x: max(x, 0)), which takes its output dtype from the
    # first hidden unit: when that is negative, max returns int 0 and every unit is truncated to int
    hidden = jnp.maximum(pre, 0.0)
    hidden = jnp.where(pre[0] < 0, jnp.trunc(hidden), hidden)
    out = hidden @ p["w_out"]
    s = jnp.stack([first0, first1, o.my_last, o.their_last, their_c, their_d, my_c, my_d])
    return jnp.where(out > 0, C, D), s.astype(jnp.int32)


def ann(name, weights, n_features=17, n_hidden=10, width=10):
    """weights: axelrod's flat layout, input-to-hidden rows, then hidden-to-output, then bias."""
    w = np.asarray(weights, np.float32)
    n_in = n_features * n_hidden
    pad = (0, width - n_hidden)
    params = {"w_in": jnp.asarray(np.pad(w[:n_in].reshape(n_hidden, n_features), (pad, (0, 0)))),
              "w_out": jnp.asarray(np.pad(w[n_in:n_in + n_hidden], pad)),
              "bias": jnp.asarray(np.pad(w[n_in + n_hidden:], pad))}
    return Bot(name, _ann_init, _ann_act, params=params, memory_depth=float("inf"))


def from_axelrod(name):
    """Reads the trained weights from axelrod's data/ann_weights.csv (lazy import)."""
    from axelrod.load_data_ import load_weights

    n_features, n_hidden, weights = load_weights()[name]
    return ann(name, weights, n_features, n_hidden)


STRATEGIES = [from_axelrod(n) for n in ["Evolved ANN", "Evolved ANN 5", "Evolved ANN 5 Noise 05"]]
