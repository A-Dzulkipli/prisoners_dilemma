"""Matches, round-robin tournaments and Moran processes, all in JAX."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ipd.game import AXELROD_PAYOFF, C, Config, Obs, check_payoff


class MatchResult(NamedTuple):
    actions: jnp.ndarray  # [steps, 2], -1 after the match ended
    scores: jnp.ndarray  # [steps, 2], 0 after the match ended
    length: jnp.ndarray  # turns actually played


class TournamentResult(NamedTuple):
    names: list
    scores: jnp.ndarray  # [n, n, reps] mean per-turn score of i against j
    cooperation: jnp.ndarray  # [n, n, reps] cooperation rate of i against j
    cfg: Config
    payoffs: jnp.ndarray  # [n, 2, 2]


def _run(init_a, act_a, init_b, act_b, payoff_a, payoff_b, key, cfg):
    k_len, k_a, k_b, k_play = jax.random.split(key, 4)
    if cfg.prob_end > 0:
        length = jnp.minimum(jax.random.geometric(k_len, cfg.prob_end), cfg.steps)
        turns_seen = -1
    else:
        length = min(cfg.turns, cfg.steps)
        turns_seen = cfg.turns
    hidden = jnp.zeros((2, 2), jnp.float32)
    b_seen_by_a = payoff_b if cfg.reveal_opponent_payoff else hidden
    a_seen_by_b = payoff_a if cfg.reveal_opponent_payoff else hidden

    def step(carry, inp):
        state_a, state_b, last_a, last_b = carry
        t, key = inp
        ka, kb, kn = jax.random.split(key, 3)
        a, state_a = act_a(state_a, Obs(last_a, last_b, t, turns_seen, payoff_a, b_seen_by_a), ka)
        b, state_b = act_b(state_b, Obs(last_b, last_a, t, turns_seen, payoff_b, a_seen_by_b), kb)
        # noise flips the played action, and both bots see the flipped one
        flip = jax.random.bernoulli(kn, cfg.noise, (2,))
        a = jnp.where(flip[0], 1 - a, a).astype(jnp.int32)
        b = jnp.where(flip[1], 1 - b, b).astype(jnp.int32)
        live = t < length
        actions = jnp.where(live, jnp.stack([a, b]), -1)
        scores = jnp.where(live, jnp.stack([payoff_a[a, b], payoff_b[b, a]]), 0.0)
        return (state_a, state_b, a, b), (actions, scores)

    carry = (init_a(k_a), init_b(k_b), jnp.int32(-1), jnp.int32(-1))
    xs = (jnp.arange(cfg.steps), jax.random.split(k_play, cfg.steps))
    _, (actions, scores) = lax.scan(step, carry, xs)
    return MatchResult(actions, scores, length)


def play_match(bot_a, bot_b, key, cfg=Config(), payoff_a=AXELROD_PAYOFF, payoff_b=AXELROD_PAYOFF):
    """One match. Pure and jittable; vmap over keys, or over bot params via bot._replace(params=...)."""
    return _run(
        lambda k: bot_a.init(bot_a.params, k),
        lambda s, o, k: bot_a.act(bot_a.params, s, o, k),
        lambda k: bot_b.init(bot_b.params, k),
        lambda s, o, k: bot_b.act(bot_b.params, s, o, k),
        jnp.asarray(payoff_a, jnp.float32),
        jnp.asarray(payoff_b, jnp.float32),
        key,
        cfg,
    )


def round_robin(bots, cfg=Config(), payoffs=None):
    """Every pair once (self-play included), cfg.reps times. payoffs: [n, 2, 2], default Axelrod's.

    Different bots are dispatched with lax.switch, so every match carries every bot's state and,
    under vmap, evaluates every bot's act. Fine for hand-written bots; vmap play_match over params
    for large families instead.
    """
    n = len(bots)
    payoffs = np.broadcast_to(AXELROD_PAYOFF if payoffs is None else payoffs, (n, 2, 2))
    for p in np.unique(payoffs, axis=0):
        check_payoff(p)
    payoffs = jnp.asarray(payoffs, jnp.float32)

    def init(key):
        return tuple(b.init(b.params, k) for b, k in zip(bots, jax.random.split(key, n)))

    def branch(i):
        def act(states, obs, key):
            a, s = bots[i].act(bots[i].params, states[i], obs, key)
            return jnp.asarray(a, jnp.int32), states[:i] + (s,) + states[i + 1 :]

        return act

    branches = [branch(i) for i in range(n)]

    def one(i, j, key):
        act_i = lambda s, o, k: lax.switch(i, branches, s, o, k)
        act_j = lambda s, o, k: lax.switch(j, branches, s, o, k)
        r = _run(init, act_i, init, act_j, payoffs[i], payoffs[j], key, cfg)
        return r.scores.sum(0) / r.length, (r.actions == C).sum(0) / r.length

    I, J = np.triu_indices(n)
    keys = jax.random.split(jax.random.key(cfg.seed), (len(I), cfg.reps))
    over_reps = jax.vmap(one, in_axes=(None, None, 0))
    score, coop = jax.jit(jax.vmap(over_reps))(I, J, keys)  # [pairs, reps, 2]

    def fill(x):  # diagonal keeps the first player's value
        return jnp.zeros((n, n, cfg.reps)).at[J, I].set(x[..., 1]).at[I, J].set(x[..., 0])

    return TournamentResult([b.name for b in bots], fill(score), fill(coop), cfg, payoffs)


def moran(scores, population, key, max_steps=100_000, mutation_rate=0.0):
    """Birth-death Moran process on expected payoffs.

    scores: [n_types, n_types] mean per-turn score, e.g. result.scores.mean(-1).
    population: [N] int type of each individual. Fitness is the mean score against everyone else,
    so unlike axelrod's MoranProcess, matches are not replayed each step. Birth is proportional
    to fitness and death is uniform among the others. Without mutation it stops at fixation.
    Returns (final population, steps taken).
    """
    n_types, N = scores.shape[0], population.shape[0]

    def fitness(pop):
        m = scores[pop][:, pop]
        return (m.sum(1) - jnp.diag(m)) / (N - 1)

    def cond(carry):
        pop, _, t = carry
        return (t < max_steps) & ((mutation_rate > 0) | jnp.any(pop != pop[0]))

    def body(carry):
        pop, key, t = carry
        key, kb, kd, km, kt = jax.random.split(key, 5)
        f = fitness(pop)
        born = jax.random.choice(kb, N, p=f / f.sum())
        dead = jax.random.randint(kd, (), 0, N - 1)
        dead = dead + (dead >= born)
        mutant = jax.random.randint(kt, (), 0, n_types - 1)
        mutant = mutant + (mutant >= pop[born])
        child = jnp.where(jax.random.bernoulli(km, mutation_rate), mutant, pop[born])
        return pop.at[dead].set(child), key, t + 1

    pop, _, steps = lax.while_loop(cond, body, (jnp.asarray(population), key, 0))
    return pop, steps
