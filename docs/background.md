# Background: IPD standards and a JAX assessment

Citations are from memory and have not been re-checked against the papers yet.

## 1. Existing standards

**Payoffs.** Standard values are T=5, R=3, P=1, S=0 (temptation, reward, punishment, sucker).
Constraints: T > R > P > S makes defection dominant in one shot, and 2R > T + S makes mutual cooperation
beat taking turns exploiting each other. Report the matrix with every result and never hard-code it.

**Axelrod's tournaments (1980).**
- First: 14 entries plus Random, round robin, 200 rounds, repeated 5 times. Tit For Tat won.
- Second: 62 entries plus Random, with a per-round end probability (about 0.00346, a median near 200 rounds),
  so nobody could exploit a known final round. TFT won again.
- His takeaways (nice, retaliatory, forgiving, clear) are the folklore baseline, and they are known to break
  under noise and in different populations.

**Axelrod Python library** (`axelrod`, Knight et al.). This is the de facto reference implementation.
- A strategy is a `Player` subclass with `strategy(self, opponent) -> Action` (C or D). It reads
  `self.history` and `opponent.history`, keeps any state on `self`, and is cleared by `reset()`.
- Each strategy carries a `classifier` dict: `memory_depth`, `stochastic`, `makes_use_of` (e.g. game, length),
  `long_run_time`, and `inspects_source` / `manipulates_source` / `manipulates_state` flags for cheaters.
  Tournaments usually exclude the cheaters.
- `Match(players, turns=200, noise=0, prob_end=None, seed=...)`.
  `Tournament(players, turns=200, repetitions=10, noise, prob_end, seed)` runs a round robin, self-play included.
  Results are ranked by mean score per turn, with wins and cooperation rates alongside.
- **Noise** flips each chosen action with probability `noise`, and the flipped move is what both histories record.
  Typical sweeps run 0 to 0.1. Results change sharply: TFT falls into retaliation spirals, while
  Win-Stay-Lose-Shift and Generous TFT hold up.
- **Moran process**: `MoranProcess(players, turns, noise, mutation_rate, mode="bd"|"db", interaction_graph=...)`.
  Fitness is the match score, and the run ends at fixation unless there is mutation.
  Fixation probability against neutral 1/N is the standard metric.
- It also offers fingerprints (Ashlock), spatial tournaments, and 200+ strategies with their original sources.

**Benchmark strategies.** AllC, AllD, Random(0.5), TFT, TF2T, Generous TFT, Grim, WSLS/Pavlov,
Alternator, Prober/Tester, Joss, Gradual, Adaptive, and the zero-determinant extortioners and generous ZD
strategies (Press & Dyson 2012, Stewart & Plotkin 2013). For named tournament reproductions, use the library's
first- and second-tournament strategy lists.

**Evolutionary and learning work.**
- Memory-one strategies are 4-vectors (p_CC, p_CD, p_DC, p_DD). For infinite or discounted games the payoff is
  exact from the stationary distribution of a 4-state Markov chain, so no simulation is needed. This underlies
  Nowak & Sigmund (GTFT 1992, WSLS 1993) and Press & Dyson's ZD result.
- Lindgren (1991) evolved growing-memory strategies.
- Harper et al. (2017, PLOS ONE) evolved and trained lookup tables, FSMs, HMMs, and small ANNs against
  the library's strategy set (axelrod-dojo). Learned strategies top standard and noisy tournaments.
- Glynatsi et al. ran very large tournament sweeps on the properties of winning strategies. Context matters
  more than any single rule.
- RL: Sandholm & Crites (1996) used Q-learning. LOLA (Foerster et al. 2018) differentiates through the
  opponent's learning step using the exact memory-one IPD value, which is a natural fit for JAX.
- LLMs: Akata et al. (2023) found GPT-4 plays unforgivingly, close to Grim. Fontana et al. (2024,
  "Nicer than humans") studied prompting effects. Both found results sensitive to prompt framing.

## 2. JAX on Colab

Yes, JAX is the right tool. On Colab it gives you a free T4/L4 GPU, and the work is plain array code.

**Where it wins**
- `lax.scan` over rounds plus `vmap` over (pair x seed x noise x payoff) runs 1e6+ matches per call.
- Bot families such as memory-n tables, FSMs, and small nets are one `act` vmapped over parameter arrays.
- Exact memory-one payoffs are a batched 4x4 linear solve, which is fast and differentiable.
- Evolutionary sweeps and gradient or RL training stay inside one jitted program.

**Remaining costs, now that every bot is your own**
- **Small runs aren't faster.** Rounds are sequential, so one 200-turn match is about 200 tiny GPU steps.
  The GPU only pays when the batch is wide, so write experiments as sweeps.
- **Recompiles.** Changing `t_max`, the bot count, or any static arg recompiles, which takes seconds
  on Colab. Keep shapes fixed and pad.
- **Mixed hand-written bots** dispatched with `lax.switch` under vmap compute every branch. Heuristic bots
  are a handful of ops each, so this barely matters below a few hundred distinct bots.
- **Unbounded memory** (e.g. "count every defection so far") becomes a counter or a `t_max` buffer
  in state. This is easy, but it has to be designed in.
- **LLM bots** are the one real exception, since they can't run inside `jit`. Run them outside the engine
  on the same `Obs`/`act` contract, or distill them into a JAX policy.
- **Colab housekeeping**: sessions get preempted, so checkpoint to Drive. Pin the jax version and
  keep the PRNG key discipline explicit.
