"""Zero-determinant (linear relation) memory-one strategies."""
from ipd.bots import memory_one
from ipd.game import AXELROD_PAYOFF

(R, S), (T, P) = AXELROD_PAYOFF.tolist()


def lr_player(name, phi=0.2, s=0.1, l=1.0):
    """Hilbe et al. (2013) eq. 14: enforces s (S_xy - l) = S_yx - l, vector from the default payoff."""
    four_vector = [
        1 - phi * (1 - s) * (R - l),
        1 - phi * (s * (l - S) + (T - l)),
        phi * ((l - S) + s * (T - l)),
        phi * (1 - s) * (l - P),
    ]
    return memory_one(four_vector, name=name)


STRATEGIES = [
    lr_player("ZD-Extort-2", 1 / 9, 0.5, P),
    lr_player("ZD-Extort-2 v2", 1 / 8, 0.5, 1),
    lr_player("ZD-Extort3", 3 / 26, 1 / 3, 1),
    lr_player("ZD-Extort-4", 4 / 17, 0.25, 1),
    lr_player("ZD-Extortion", 0.2, 0.1, 1),
    lr_player("ZD-GTFT-2", 0.25, 0.5, R),
    lr_player("ZD-GEN-2", 1 / 8, 0.5, 3),
    lr_player("ZD-Mischief", 0.1, 0.0, 1),
    lr_player("ZD-SET-2", 1 / 4, 0.0, 2),
]
