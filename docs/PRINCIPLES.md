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
- 2026-10-10: All 36 first- and second-tournament strategies are ported and match the axelrod library
  (deterministic ones move for move). Mean tournament scores agree within 0.01 per turn. Neither the
  library nor our port reproduces the published rankings: Stein & Rapoport and Grofman top the first
  tournament instead of TFT, because the reconstructions fill gaps in the 1980 descriptions
  (`scripts/axelrod_tournaments.py`).
- 2026-10-10: Several library strategies (DBS, Revised Downing, Math Constant Hunter, Adaptive TFT,
  Momentum) depend on float64 rounding at exact ties. The float32 ports carry precomputed tie tables or
  float32 pairs to match. DBS still differs from the library on about 3% of sequences, where the
  library breaks mathematically exact ties by rounding noise.
- 2026-10-10: Move-for-move checks run without noise and with a known length, so noise-specific paths
  (e.g. Contrite TFT) and the unknown-length path are only smoke-tested.
