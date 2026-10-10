# Principles and learnings

## Principles
1. **Comparable to the literature.** Default to Axelrod conventions: payoffs (5,3,1,0) for every bot,
   200 turns, round robin with self-play, and mean score per turn. Any deviation, including per-bot
   payoffs, is stated in the result.
2. **Every result is reproducible from a config.** Payoffs, turns or prob_end, noise, repetitions,
   seed, bot set, and code commit are all recorded with the output.
3. **JAX all the way down.** Every bot is a pure function over fixed-size state, so the whole pipeline
   jits and vmaps. Batch width is where the GPU pays, so experiments are written as sweeps.
4. **`axelrod` is the oracle.** Port its strategies when needed, and check against its results in tests.
5. **Simple over clever.** Research code is read more than it is run. Prefer short functions and
   explicit arrays to frameworks and abstractions.
6. **Noise is not optional.** Report noiseless and noisy results (e.g. 0, 0.01, 0.05), because rankings flip.
7. **Report spread.** Give the mean with an interval over seeds or repetitions, never a single run.

## Learnings
Findings from experiments. Owner-requested notes and future interests live in `NOTES.md`.
Add one dated line per finding, with a link to the notebook or commit that shows it.

- 2026-10-10: Project started. Background summary in `docs/background.md`.
