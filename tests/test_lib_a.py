import axelrod as axl
import numpy as np
import pytest
from axelrod.mock_player import MockPlayer
from axl_check import KEY, TO_AXL, assert_same_distribution, assert_same_moves, sequence_bot

from ipd.engine import play_match
from ipd.game import Config
from ipd.strategies import forgiver, grudger, oncebitten, punisher, resurrection, retaliate, titfortat

STRATEGIES = (
    titfortat.STRATEGIES
    + grudger.STRATEGIES
    + retaliate.STRATEGIES
    + punisher.STRATEGIES
    + oncebitten.STRATEGIES
    + forgiver.STRATEGIES
    + resurrection.STRATEGIES
)
AXL = {s.name: s for s in axl.all_strategies}


@pytest.mark.parametrize("bot", STRATEGIES, ids=lambda b: b.name)
def test_matches_axelrod(bot):
    assert bot.stochastic == AXL[bot.name].classifier["stochastic"]
    if bot.stochastic:
        assert_same_distribution(bot, AXL[bot.name])
    else:
        assert_same_moves(bot, AXL[bot.name])


@pytest.mark.parametrize("prefix", [[], [1], [0, 1], [1, 1], [0, 1, 0, 1], [1, 0, 1, 1, 1, 0]])
def test_adaptive_tft_float64_saturation(prefix):
    # long C runs push the library's float64 world to exactly 1.0, after which a single D is forgiven
    seq = list(prefix)
    for run in (52, 53, 54, 55, 1, 53, 54):
        seq += [0] * run + [1, 1]
    turns = len(seq)
    ours = play_match(titfortat.adaptive_tit_for_tat, sequence_bot(seq), KEY, Config(turns=turns)).actions[:, 0]
    history = axl.Match((axl.AdaptiveTitForTat(), MockPlayer([TO_AXL[a] for a in seq])), turns=turns).play()
    theirs = np.array([int(a == axl.Action.D) for a, _ in history])
    np.testing.assert_array_equal(np.asarray(ours), theirs)
