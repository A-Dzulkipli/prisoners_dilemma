import sys

import axelrod as axl
import pytest
from axl_check import assert_same_distribution, assert_same_moves

from ipd.strategies import (
    adaptive, adaptor, apavlov, appeaser, averagecopier, backstabber, better_and_better, calculator, cooperator, dbs,
    defector, doubler, frequency_analyzer, gradualkiller, grumpy, handshake, hunter, inverse, momentum, negation,
    revised_downing, selfsteem, shortmem, stalker, verybad,
)

MODULES = [
    apavlov, adaptive, adaptor, hunter, appeaser, averagecopier, backstabber, better_and_better, calculator, dbs,
    doubler, frequency_analyzer, gradualkiller, grumpy, handshake, inverse, momentum, negation, revised_downing,
    selfsteem, shortmem, stalker, cooperator, defector, verybad,
]
STRATEGIES = [bot for m in MODULES for bot in m.STRATEGIES]
AXL = {s.name: s for s in axl.all_strategies}
# a defection run that ends the match is geometric with mean ~100 turns, so 300 seeds are too few
N_MATCHES = {"AdaptorBrief": 1000, "AdaptorLong": 1000}
DBS_TIES = (
    "DBS's tree search has exact ties (equal values in exact arithmetic) that the library breaks by "
    "float64 rounding noise rather than its documented C-on-equal rule; see test_dbs_with_exact_ties"
)


def _param(bot):
    return pytest.param(bot, id=bot.name, marks=pytest.mark.xfail(reason=DBS_TIES) if bot.name == "DBS" else ())


@pytest.mark.parametrize("bot", [_param(b) for b in STRATEGIES])
def test_matches_axelrod(bot):
    axl_cls = AXL[bot.name]
    assert bot.stochastic == axl_cls.classifier["stochastic"]
    if bot.stochastic:
        assert_same_distribution(bot, axl_cls, n=N_MATCHES.get(bot.name, 300))
    else:
        assert_same_moves(bot, axl_cls)


def test_dbs_with_exact_ties(monkeypatch):
    """DBS matches the library once the library's tree-search values within 1e-9 count as ties (-> C)."""
    lib = sys.modules["axelrod.strategies.dbs"]

    def move_gen(outcome, policy, depth_search_tree=5):
        node = lib.DeterministicNode(outcome[0], outcome[1], depth=0)
        value_c, value_d = lib.minimax_tree_search(node, policy, depth_search_tree)
        return axl.Action.C if value_c >= value_d - 1e-9 else axl.Action.D

    monkeypatch.setattr(lib, "move_gen", move_gen)
    assert_same_moves(dbs.STRATEGIES[0], axl.DBS, n_seqs=200)
