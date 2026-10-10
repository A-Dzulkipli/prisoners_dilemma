"""Negation."""
import jax.numpy as jnp

from ipd.bots import Bot, coop_with_prob, no_state

negation = Bot(
    "Negation",
    no_state,
    lambda p, s, o, k: (jnp.where(o.t == 0, coop_with_prob(k, 0.5), 1 - o.their_last), s),
    stochastic=True,
)

STRATEGIES = [negation]
