# Rules for LLM contributors

**Scope**
- Take one bounded task: one bot, one experiment, or one engine function. Don't expand the scope
  or refactor nearby code unasked.
- Read `PRINCIPLES.md` and `CONTRACT.md` first. If the task conflicts with either, stop and say so.

**Code style**
- Write plain researcher-style code: short functions, numpy/jax arrays, descriptive names, and comments only
  where the reason isn't obvious.
- No defensive bloat. Skip validation of internal arguments, try/except around code that shouldn't fail,
  config systems, class hierarchies, logging frameworks, and backwards-compat shims.
- Match the surrounding code. Don't add dependencies without asking.
- A new bot comes with one test against a known opponent sequence (e.g. TFT vs Alternator).

**Outputs**
- Commit code, configs, small summary tables (CSV under ~100 KB), and figures referenced in docs.
- Don't commit raw match logs, large arrays, checkpoints, LLM response caches, or notebook outputs.
  These go under `data/` or `runs/`, which are gitignored.
- Clear notebook outputs before committing (`nbstripout`), unless the notebook is the published result.

**Results**
- State the cfg (payoff, turns, noise, reps, seed) and the commit with every claimed result.
- Add a one-line entry to the Learnings section of `PRINCIPLES.md` when a finding changes what we believe.
