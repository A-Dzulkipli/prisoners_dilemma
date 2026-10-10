import sys

import axelrod as axl
import pytest
from axl_check import AXL, assert_same_distribution, assert_same_moves

from ipd.strategies import all_strategies, dbs, meta, titfortat

EXCLUDED = {"Darwin"}  # inspects and manipulates its opponent
# metas have their own checks, some marked slow, in test_lib_meta.py
BOTS = [b for b in all_strategies() if b.name not in {m.name for m in meta.STRATEGIES}]
# ALLCorALLD self-play scores are all-or-nothing per match (3, 1, 0 or 5), and an Adaptor defection run
# that ends the match is geometric with mean ~100 turns, so their means need more matches
N_MATCHES = {"ALLCorALLD": 2000, "AdaptorBrief": 1000, "AdaptorLong": 1000}
# a 24-layer transformer per move: keep it short
SHORT = {"EvolvedAttention": dict(turns=40, n_seqs=6)}
DBS_TIES = (
    "DBS's tree search has exact ties (equal values in exact arithmetic) that the library breaks by "
    "float64 rounding noise rather than its documented C-on-equal rule; see test_dbs_with_exact_ties"
)


def test_every_library_strategy_is_ported_once():
    names = [b.name for b in all_strategies()]
    assert len(names) == len(set(names))
    assert {s.name for s in axl.all_strategies} - EXCLUDED - set(names) == set()


def _param(bot):
    return pytest.param(bot, id=bot.name, marks=pytest.mark.skip(reason=DBS_TIES) if bot.name == "DBS" else ())


@pytest.mark.parametrize("bot", [_param(b) for b in BOTS])
def test_matches_axelrod(bot):
    axl_cls = AXL[bot.name]
    assert bot.stochastic == axl.Classifiers["stochastic"](axl_cls())
    if bot.stochastic:
        assert_same_distribution(bot, axl_cls, n=N_MATCHES.get(bot.name, 300))
    else:
        assert_same_moves(bot, axl_cls, **SHORT.get(bot.name, {}))


@pytest.mark.parametrize("prefix", [[], [1], [0, 1], [1, 1], [0, 1, 0, 1], [1, 0, 1, 1, 1, 0]])
def test_adaptive_tft_float64_saturation(prefix):
    # long C runs push the library's float64 world to exactly 1.0, after which a single D is forgiven
    seq = list(prefix)
    for run in (52, 53, 54, 55, 1, 53, 54):
        seq += [0] * run + [1, 1]
    assert_same_moves(titfortat.adaptive_tit_for_tat, axl.AdaptiveTitForTat, turns=len(seq), seqs=[seq])


def test_dbs_with_exact_ties(monkeypatch):
    """DBS matches the library once the library's tree-search values within 1e-9 count as ties (-> C)."""
    lib = sys.modules["axelrod.strategies.dbs"]

    def move_gen(outcome, policy, depth_search_tree=5):
        node = lib.DeterministicNode(outcome[0], outcome[1], depth=0)
        value_c, value_d = lib.minimax_tree_search(node, policy, depth_search_tree)
        return axl.Action.C if value_c >= value_d - 1e-9 else axl.Action.D

    monkeypatch.setattr(lib, "move_gen", move_gen)
    assert_same_moves(dbs.STRATEGIES[0], axl.DBS)
