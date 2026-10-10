# Simulator and bot contract

The simulator is JAX: one engine, jit/vmap-able end to end.

## Game
- Actions are integers: `C = 0`, `D = 1`.
- **Payoffs are per bot.** Each bot is scored by its own 2x2 matrix, `payoff[i][my_action, their_action]`.
  A tournament takes `payoffs` with shape `[n_bots, 2, 2]`. The default is Axelrod's `[[3, 0], [5, 1]]`
  broadcast to every bot, so the symmetric game is just the special case where every row is the same.
- Bots can read their own matrix (it is in `Obs`). Whether a bot can see its opponent's matrix is a
  tournament setting, `reveal_opponent_payoff`, which defaults to False.
- The engine checks T > R > P > S and 2R > T + S per matrix, and warns rather than raises, since
  non-PD matrices are a legitimate experiment.
- A match runs a fixed `turns`. With `prob_end`, the length is sampled once per match, and the match
  runs to `T_max` with a mask.
- Noise: after both bots act, each action flips independently with probability `noise`. Both bots
  then observe the **flipped** actions, as in `axelrod`.

## Bot
```python
class Bot(NamedTuple):
    name: str
    init: Callable[[params, key], State]                     # State is a pytree of fixed-shape arrays
    act:  Callable[[params, State, Obs, key], tuple[int, State]]
```
- `Obs` holds:
  - `my_last` and `their_last`: ints, -1 on round 0
  - `t`: the round index
  - `turns`: the match length, or -1 if unknown
  - `my_payoff`: the bot's own matrix
  - `their_payoff`: the opponent's matrix, or zeros when it is not revealed

  Bots that need more history keep it in `State` as a fixed-length buffer (up to `T_max`).
- No Python control flow on traced values (use `jnp.where` / `lax.cond`), no I/O, no globals.
  All randomness comes from `key`.
- `params` is an array pytree, so a bot family is one `act` plus many parameter vectors.
- Each bot declares `memory_depth` and `stochastic` in the axelrod classifier style.

## Engine
- `play_match(bot_a, bot_b, payoff_a, payoff_b, cfg) -> MatchResult` returns `actions [T, 2]`
  and `scores [T, 2]`.
- `round_robin(bots, payoffs, cfg) -> TournamentResult` returns a mean score-per-turn matrix of
  `[n, n, reps]`, plus cooperation rates.
- `moran(bots, payoffs, cfg)` and the evolutionary routines are built on the score matrix and do not
  replay matches when the matrix can be reused.
- `cfg` is one frozen dataclass: turns, prob_end, T_max, noise, reps, seed, and reveal_opponent_payoff.
  It is saved with every result, along with `payoffs`.
- Hand-written bots that differ in code are dispatched with `lax.switch` on a bot index. Bot families
  that share an `act` are vmapped over `params`.

## Validation
- `tests/` checks known sequences (e.g. TFT vs Alternator) and reproduces a few `axelrod` library
  results under default payoffs, with `axelrod` used as a test-time oracle only.
