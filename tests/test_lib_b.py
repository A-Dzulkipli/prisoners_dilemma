import axelrod as axl
import pytest
from axl_check import assert_same_distribution, assert_same_moves

from ipd.strategies import (
    cycler,
    gobymajority,
    mathematicalconstants,
    memoryone,
    memorytwo,
    mutual,
    prober,
    sequence_player,
    worse_and_worse,
    zero_determinant,
)

AXL = {s.name: s for s in axl.all_strategies}
MODULES = [
    memoryone, zero_determinant, memorytwo, gobymajority, prober, cycler, sequence_player,
    mathematicalconstants, worse_and_worse, mutual,
]
BOTS = [b for m in MODULES for b in m.STRATEGIES]
# ALLCorALLD self-play scores are all-or-nothing per match (3, 1, 0 or 5), so the mean needs more samples
N_SAMPLES = {"ALLCorALLD": 2000}


def test_names_unique():
    assert len({b.name for b in BOTS}) == len(BOTS) == 56


@pytest.mark.parametrize("bot", BOTS, ids=lambda b: b.name)
def test_matches_axelrod(bot):
    axl_cls = AXL[bot.name]
    assert bot.stochastic == axl.Classifiers["stochastic"](axl_cls())
    if bot.stochastic:
        assert_same_distribution(bot, axl_cls, n=N_SAMPLES.get(bot.name, 300))
    else:
        assert_same_moves(bot, axl_cls)
