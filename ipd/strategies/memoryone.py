"""Memory-one strategies, ported from axelrod's memoryone.py (GTFT and WSLS live in ipd.bots)."""
from ipd.bots import memory_one

win_shift_lose_stay = memory_one([0, 1, 1, 0], p0=0.0, name="Win-Shift Lose-Stay")._replace(stochastic=False)
firm_but_fair = memory_one([1, 0, 1, 2 / 3], name="Firm But Fair")
stochastic_cooperator = memory_one([0.935, 0.229, 0.266, 0.42], name="Stochastic Cooperator")


def stochastic_wsls(ep=0.05):
    return memory_one([1 - ep, ep, ep, 1 - ep], name="Stochastic WSLS")


def soft_joss(q=0.9):
    return memory_one([1, 1 - q, 1, 1 - q], name="Soft Joss")


# repeats its own last move after a random first move (C with probability 0.6)
all_c_or_all_d = memory_one([1, 1, 0, 0], p0=0.6, name="ALLCorALLD")

STRATEGIES = [
    all_c_or_all_d,
    firm_but_fair,
    soft_joss(),
    stochastic_cooperator,
    stochastic_wsls(),
    win_shift_lose_stay,
]
