"""Gambler strategies (stochastic lookup tables), ported from axelrod's gambler.py."""
from ipd.strategies.lookerup import lookup


def pso_gambler(name, depths):
    """Reads the PSO-trained table from axelrod's data/pso_gambler.csv (lazy import)."""
    from axelrod.load_data_ import load_pso_tables

    pattern = load_pso_tables("pso_gambler.csv", directory="data")[(name,) + depths]
    return lookup(name, pattern, depths, stochastic=True)


zd_mem2 = lookup(
    "ZD-Mem2",
    [11 / 12, 4 / 11, 7 / 9, 1 / 10, 5 / 6, 3 / 11, 7 / 9, 1 / 10,
     2 / 3, 1 / 11, 7 / 9, 1 / 10, 3 / 4, 2 / 11, 7 / 9, 1 / 10],
    (2, 2, 0),
    stochastic=True,
)

STRATEGIES = [
    pso_gambler("PSO Gambler 1_1_1", (1, 1, 1)),
    pso_gambler("PSO Gambler 2_2_2", (2, 2, 2)),
    pso_gambler("PSO Gambler 2_2_2 Noise 05", (2, 2, 2)),
    pso_gambler("PSO Gambler Mem1", (1, 1, 0)),
    zd_mem2,
]
