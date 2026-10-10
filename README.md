# prisoners_dilemma

Iterated prisoner's dilemma experiments. A JAX simulator, with bots as pure functions over fixed-size state.
Read `docs/PRINCIPLES.md`, `docs/CONTRACT.md`, `docs/LLM_CONTRIBUTORS.md` and `docs/NOTES.md` first.

```python
# Colab: !pip install git+https://github.com/A-Dzulkipli/prisoners_dilemma
from ipd import Config, round_robin
from ipd import bots

result = round_robin(bots.CLASSICS + [bots.generous_tit_for_tat()], Config(noise=0.01, reps=20))
```

`ipd.strategies.all_strategies()` returns every ported axelrod library strategy, each verified against
the library. It needs the `library` extra (`pip install ".[library]"`): the FSM, ANN, gambler and attention
modules read their tables and weights from `axelrod` at import, and attention also needs `torch`.
The meta strategies run teams of up to 220 bots, about 3 s per 200-turn match when the team includes
EvolvedAttention.

Tests check the engine and every strategy against the `axelrod` library: `pip install -e ".[test]" && pytest`.
The big-team meta checks are marked slow (~45 min); run them with `pytest -m slow`.
