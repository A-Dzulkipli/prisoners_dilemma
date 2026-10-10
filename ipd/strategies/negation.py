"""Negation, ported from axelrod's negation.py."""
import jax.numpy as jnp

from ipd.bots import Bot, _coop_with_prob, no_state

negation = Bot("Negation", no_state, lambda p, s, o, k: (jnp.where(o.t == 0, _coop_with_prob(k, 0.5), 1 - o.their_last), s),
               memory_depth=1, stochastic=True)

STRATEGIES = [negation]
