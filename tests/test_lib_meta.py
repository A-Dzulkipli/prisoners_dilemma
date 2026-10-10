import axelrod as axl
import pytest
from axelrod.classifier import Classifiers
from axl_check import assert_same_distribution, assert_same_moves

from ipd.strategies import meta

AXL = {s.name: s for s in axl.all_strategies}
# small teams get the full checks; ~37-220 member teams are slow on both sides, so shorter, fewer matches
SMALL = {"Memory Decay", "Meta Hunter", "Meta Hunter Aggressive"}
# vs Suspicious TFT the outcome is bimodal, and Mixer's per-turn draws need more matches than 30
N_MATCHES = {"Meta Majority Memory One": 300, "Meta Mixer": 100}


def _param(bot):
    return pytest.param(bot, id=bot.name, marks=() if bot.name in SMALL else pytest.mark.slow)


@pytest.mark.parametrize("bot", [_param(b) for b in meta.STRATEGIES])
def test_matches_axelrod(bot):
    axl_cls = AXL[bot.name]
    # metas set stochastic from their team at init, so ask an instance rather than the class
    assert bot.stochastic == Classifiers["stochastic"](axl_cls())
    if bot.name in SMALL:
        check = assert_same_distribution if bot.stochastic else assert_same_moves
        check(bot, axl_cls)
    elif bot.stochastic:
        assert_same_distribution(bot, axl_cls, turns=40, n=N_MATCHES.get(bot.name, 30))
    else:
        assert_same_moves(bot, axl_cls, turns=40, n_seqs=4)
