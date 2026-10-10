import axelrod as axl
import pytest
from axl_check import assert_same_distribution, assert_same_moves

from ipd.strategies.axelrod_first import STRATEGIES

AXL = {s.name: s for s in axl.all_strategies}


@pytest.mark.parametrize("bot", STRATEGIES, ids=lambda b: b.name)
def test_matches_axelrod(bot):
    if bot.stochastic:
        assert_same_distribution(bot, AXL[bot.name])
    else:
        assert_same_moves(bot, AXL[bot.name])
