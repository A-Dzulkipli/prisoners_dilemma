import axelrod as axl

from ipd.strategies import all_strategies

EXCLUDED = {"Darwin"}  # inspects and manipulates its opponent


def test_every_library_strategy_is_ported_once():
    names = [b.name for b in all_strategies()]
    assert len(names) == len(set(names))
    assert {s.name for s in axl.all_strategies} - EXCLUDED - set(names) == set()
