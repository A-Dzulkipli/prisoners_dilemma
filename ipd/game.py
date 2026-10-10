"""Actions, payoffs, match config and what a bot observes each round."""
import warnings
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

C, D = 0, 1

# payoff[my_action, their_action]: R=3, S=0, T=5, P=1
AXELROD_PAYOFF = np.array([[3.0, 0.0], [5.0, 1.0]], dtype=np.float32)


def check_payoff(payoff):
    """Warn (not raise) if a 2x2 matrix is not a prisoner's dilemma."""
    (R, S), (T, P) = np.asarray(payoff)
    if not (T > R > P > S and 2 * R > T + S):
        warnings.warn(f"not a PD payoff: T={T}, R={R}, P={P}, S={S}")


@dataclass(frozen=True)
class Config:
    turns: int = 200
    prob_end: float = 0.0  # > 0: end after each turn with this probability
    t_max: int | None = None  # scan length, defaults to turns; set it when using prob_end
    noise: float = 0.0  # probability each played action is flipped
    reps: int = 10
    seed: int = 0
    reveal_opponent_payoff: bool = False

    @property
    def steps(self):
        return self.t_max or self.turns


class Obs(NamedTuple):
    my_last: jnp.ndarray  # -1 on round 0
    their_last: jnp.ndarray
    t: jnp.ndarray  # round index
    turns: jnp.ndarray  # match length, -1 if bots aren't told
    my_payoff: jnp.ndarray  # [2, 2]
    their_payoff: jnp.ndarray  # [2, 2], zeros unless revealed
