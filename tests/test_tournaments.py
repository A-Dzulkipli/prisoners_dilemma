import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import axelrod_tournaments as at  # noqa: E402


@pytest.mark.parametrize("players", [at.FIRST, at.SECOND], ids=["first", "second"])
def test_tournament_scores_match_library(players):
    ours, theirs = at.ours(players, 50), at.library(players, 10)
    for name in theirs:
        assert abs(ours[name] - theirs[name]) < 0.04, f"{name}: {ours[name]:.3f} vs {theirs[name]:.3f}"
