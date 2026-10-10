"""Thue-Morse sequence players."""
import jax.numpy as jnp
from jax import lax

from ipd.bots import INF, Bot, no_state
from ipd.game import C, D


def _thue_morse(t):
    return lax.population_count(t) % 2


thue_morse = Bot(
    "ThueMorse", no_state, lambda p, s, o, k: (jnp.where(_thue_morse(o.t) == 0, D, C), s), memory_depth=INF
)
thue_morse_inverse = Bot(
    "ThueMorseInverse", no_state, lambda p, s, o, k: (jnp.where(_thue_morse(o.t) == 0, C, D), s),
    memory_depth=INF,
)

STRATEGIES = [thue_morse, thue_morse_inverse]
