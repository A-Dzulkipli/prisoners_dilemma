"""Q-learning strategies.

The library's state string is their last 12 moves plus their total number of cooperations (an
integer count, although the docstring calls it a proportion). The count only grows, and it grows
exactly when they cooperate, so states sharing a count form a chain indexed by their trailing run
of defections (capped at 12), and a state is never revisited once they cooperate again. A [13, 2]
Q table for the current chain is therefore exact: it is reset to zeros whenever they cooperate,
since the update the library makes to the previous chain's state is never read again.
"""
import jax
import jax.numpy as jnp

from ipd.bots import INF, Bot, coop_with_prob
from ipd.game import C

MEMORY = 12


def _q_init(p, k):
    z = jnp.int32(0)
    return dict(Q=jnp.zeros((MEMORY + 1, 2), jnp.float32), V=jnp.zeros(MEMORY + 1, jnp.float32), run=z,
                prev_action=z)


def _q_act(p, s, o, k):
    lr, disc = p
    k1, k2 = jax.random.split(k)
    reset = (o.t == 0) | (o.their_last == C)
    run = jnp.where(reset, 0, jnp.minimum(s["run"] + 1, MEMORY))
    prev, a = s["run"], s["prev_action"]
    # library's reward pairs my intended (pre-noise) move with their played one
    reward = o.my_payoff[a, jnp.maximum(o.their_last, 0)]
    Q = s["Q"].at[prev, a].set((1 - lr) * s["Q"][prev, a] + lr * (reward + disc * s["V"][run]))
    V = s["V"].at[prev].set(Q[prev].max())
    Q, V = jnp.where(reset, 0.0, Q), jnp.where(reset, 0.0, V)
    greedy = jnp.where(Q[run, C] >= Q[run, 1 - C], C, 1 - C)
    action = jnp.where(jax.random.uniform(k1) < 0.9, greedy, coop_with_prob(k2, 0.5))
    return action, dict(Q=Q, V=V, run=run, prev_action=action)


def q_learner(name, learning_rate, discount_rate):
    return Bot(name, _q_init, _q_act, params=jnp.array([learning_rate, discount_rate], jnp.float32),
               memory_depth=INF, stochastic=True)


STRATEGIES = [
    q_learner("Risky QLearner", 0.9, 0.9),
    q_learner("Arrogant QLearner", 0.9, 0.1),
    q_learner("Hesitant QLearner", 0.1, 0.9),
    q_learner("Cautious QLearner", 0.1, 0.1),
]
