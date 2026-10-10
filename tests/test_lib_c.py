import axelrod as axl
import pytest
from axl_check import assert_same_distribution, assert_same_moves

from ipd.strategies import ann, attention, bush_mosteller, finite_state_machines, gambler, hmm, lookerup, qlearner

AXL = {s.name: s for s in axl.all_strategies}
BOTS = [
    b
    for m in [finite_state_machines, lookerup, gambler, hmm, ann, qlearner, bush_mosteller, attention]
    for b in m.STRATEGIES
]


@pytest.mark.parametrize("bot", BOTS, ids=lambda b: b.name)
def test_matches_axelrod(bot):
    assert bot.stochastic == AXL[bot.name].classifier["stochastic"]
    if bot.stochastic:
        assert_same_distribution(bot, AXL[bot.name])
    elif bot.name == "EvolvedAttention":  # a 24-layer transformer per move: keep it short
        assert_same_moves(bot, AXL[bot.name], turns=40, n_seqs=6)
    else:
        assert_same_moves(bot, AXL[bot.name])
