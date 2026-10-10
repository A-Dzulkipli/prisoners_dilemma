# Rules for LLM contributors

**Scope**
- Take one bounded task: one bot, one experiment, or one engine function. Don't expand the scope
  or refactor nearby code unasked.
- Read `PRINCIPLES.md`, `CONTRACT.md` and `NOTES.md` first. If the task conflicts with either, stop and say so.

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

**Git**
- The LLM is the main interface to git and GitHub, but it carries out tasks and doesn't make repo
  decisions on its own.
- Commit and push only what the task covers. Use small commits with a plain one-line subject, and
  stage files by name (no `git add -A` sweeping in generated output).
- Work on `main` unless the owner asks for a branch.
- Leave history rewrites (force push, rebase of pushed commits, amend after push), branch deletion,
  tags and releases, and repo settings to the owner. Ask first.

**Results**
- State the cfg (payoff, turns, noise, reps, seed) and the commit with every claimed result.
- Add a one-line entry to the Learnings section of `PRINCIPLES.md` when a finding changes what we believe.

**Notes**
- `NOTES.md` is a dated log of things the owner asks to have noted. When asked to note something,
  append it under today's date heading (oldest first) and don't edit past entries.
- Don't add your own entries unasked; findings go in the Learnings log in `PRINCIPLES.md`.
