import axelrod as axl
import pytest
from axl_check import assert_same_distribution, assert_same_moves

from ipd.strategies.axelrod_second_a import STRATEGIES

AXL = {s.name: s for s in axl.all_strategies}


@pytest.mark.parametrize("bot", STRATEGIES, ids=lambda b: b.name)
def test_matches_axelrod(bot):
    axl_cls = AXL[bot.name]
    assert bot.stochastic == axl_cls.classifier["stochastic"]
    if bot.stochastic:
        assert_same_distribution(bot, axl_cls)
    else:
        assert_same_moves(bot, axl_cls)
