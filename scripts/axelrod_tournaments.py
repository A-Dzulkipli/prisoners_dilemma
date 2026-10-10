"""Reproduce Axelrod's tournaments and compare with the axelrod library and the published rankings.

Run: python scripts/axelrod_tournaments.py  (needs axelrod installed)
"""
import re
import warnings

import axelrod as axl
import numpy as np

from ipd import bots
from ipd.engine import round_robin
from ipd.game import Config
from ipd.strategies import axelrod_first, axelrod_second_a, axelrod_second_b

warnings.simplefilter("ignore")
LIBRARY = {s.name: s for s in axl.all_strategies}

FIRST = axelrod_first.STRATEGIES + [bots.tit_for_tat, bots.grudger, bots.random_bot()]
SECOND = axelrod_second_a.STRATEGIES + axelrod_second_b.STRATEGIES + [bots.tit_for_tat, bots.random_bot()]

# Axelrod (1980a) final ranking of the first tournament
FIRST_PUBLISHED = {
    "Tit For Tat": 1, "First by Tideman and Chieruzzi": 2, "First by Nydegger": 3, "First by Grofman": 4,
    "First by Shubik": 5, "First by Stein and Rapoport": 6, "Grudger": 7, "First by Davis": 8,
    "First by Graaskamp": 9, "First by Downing": 10, "First by Feld": 11, "First by Joss": 12,
    "First by Tullock": 13, "First by Anonymous": 14, "Random": 15,
}
ORDINALS = {w: i + 1 for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
    "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())}


def second_published():
    """Ranks out of 63 in Axelrod (1980b), parsed from the library docstrings (TFT won)."""
    ranks = {"Tit For Tat": 1}
    for b in SECOND:
        if not b.name.startswith("Second by"):
            continue
        m = re.search(r"came (?:in )?(\d+|[a-z]+)", LIBRARY[b.name].__doc__ or "")
        if m:
            ranks[b.name] = int(re.sub(r"\D", "", m.group(1)) or 0) or ORDINALS.get(m.group(1))
    return ranks


def ours(players, reps):
    r = round_robin(players, Config(turns=200, reps=reps))
    return dict(zip(r.names, np.asarray(r.scores.mean((1, 2)))))


def library(players, reps):
    result = axl.Tournament([LIBRARY[b.name]() for b in players], turns=200, repetitions=reps, seed=0)
    return dict(zip([b.name for b in players], np.mean(result.play(progress_bar=False).payoff_matrix, axis=1)))


def ranks(scores):
    return {n: i + 1 for i, n in enumerate(sorted(scores, key=lambda n: -scores[n]))}


def table(players, published):
    a, b = ours(players, 100), library(players, 20)
    ra, rb = ranks(a), ranks(b)
    print("| Strategy | published rank | library rank | our rank | library score | our score |")
    print("|---|---|---|---|---|---|")
    for name in sorted(a, key=lambda n: published.get(n, 99)):
        print(f"| {name} | {published.get(name, '')} | {rb[name]} | {ra[name]} | {b[name]:.3f} | {a[name]:.3f} |")


if __name__ == "__main__":
    print("First tournament (all 15 entries, 200 turns)\n")
    table(FIRST, FIRST_PUBLISHED)
    print("\nSecond tournament (26 of 63 entries, 200 turns; published ranks are out of 63)\n")
    table(SECOND, second_published())
