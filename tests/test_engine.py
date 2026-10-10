import axelrod as axl
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from axl_check import CLASSIC_AXL, KEY

from ipd import bots
from ipd.engine import moran, play_match, round_robin
from ipd.game import C, D, Config


def test_tournament_matches_axelrod():
    ours = round_robin([b for b, _ in CLASSIC_AXL], Config(reps=1)).scores[:, :, 0]
    players = [cls() for _, cls in CLASSIC_AXL]
    theirs = axl.Tournament(players, turns=200, repetitions=1).play(progress_bar=False).payoff_matrix
    np.testing.assert_allclose(ours, theirs, atol=1e-5)


def test_memory_one_reproduces_named_bots():
    for vec, named in [([1, 0, 1, 0], bots.tit_for_tat), ([1, 0, 0, 1], bots.win_stay_lose_shift)]:
        for opp in bots.CLASSICS:
            m = play_match(bots.memory_one(vec), opp, KEY).actions
            np.testing.assert_array_equal(m, play_match(named, opp, KEY).actions)


def test_gtft_forgives_a_third_of_the_time():
    keys = jax.random.split(KEY, 200)
    r = jax.vmap(lambda k: play_match(bots.generous_tit_for_tat(), bots.defector, k))(keys)
    assert abs((r.actions[:, 1:, 0] == C).mean() - 1 / 3) < 0.01


def test_noise_matches_axelrod():
    cfg = Config(noise=0.05, reps=300)
    ours = round_robin([bots.tit_for_tat], cfg).scores.mean()
    players = [axl.TitForTat()]
    theirs = axl.Tournament(players, turns=200, repetitions=300, noise=0.05, seed=1)
    theirs = np.mean(theirs.play(progress_bar=False).payoff_matrix)
    assert abs(ours - theirs) < 0.05


def test_prob_end_mean_length():
    cfg = Config(prob_end=0.05, t_max=500)
    lengths = jax.vmap(lambda k: play_match(bots.cooperator, bots.cooperator, k, cfg).length)(
        jax.random.split(KEY, 2000)
    )
    assert abs(lengths.mean() - 20) < 1.5
    assert play_match(bots.tit_for_tat, bots.tit_for_tat, KEY, cfg).actions[lengths.max() :].max() == -1


def test_each_bot_scored_by_its_own_payoff():
    harsh = np.array([[3.0, -2.0], [4.0, 0.0]])
    r = play_match(bots.defector, bots.cooperator, KEY, payoff_b=harsh)
    assert r.scores.mean(0).tolist() == [5.0, -2.0]


def test_revealed_payoff_is_visible():
    peek = bots.Bot("Peek", bots.no_state, lambda p, s, o, k: (jnp.where(o.their_payoff[1, 0] > 4, D, C), s))
    for reveal, expected in [(False, C), (True, D)]:
        r = play_match(peek, bots.cooperator, KEY, Config(turns=3, reveal_opponent_payoff=reveal))
        assert (r.actions[:, 0] == expected).all()


def test_non_pd_payoff_warns():
    with pytest.warns(UserWarning):
        round_robin([bots.cooperator], Config(turns=2, reps=1), payoffs=[[[1.0, 0.0], [0.0, 1.0]]])


def test_neutral_moran_fixation_is_one_over_n():
    N = 10
    pop = jnp.array([1] + [0] * (N - 1))
    final, _ = jax.vmap(lambda k: moran(jnp.ones((2, 2)), pop, k))(jax.random.split(KEY, 4000))
    assert abs((final[:, 0] == 1).mean() - 1 / N) < 0.015


def test_defector_invades_cooperators_more_than_neutral():
    s = round_robin([bots.cooperator, bots.defector], Config(reps=1)).scores.mean(-1)
    pop = jnp.array([1, 0, 0, 0, 0])
    final, _ = jax.vmap(lambda k: moran(s, pop, k))(jax.random.split(KEY, 2000))
    assert (final[:, 0] == 1).mean() > 0.4  # neutral would be 0.2
