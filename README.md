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
the library. A few constructors read weights from `axelrod` (`pip install ".[library]"`).

Tests check the engine against the `axelrod` library: `pip install -e ".[test]" && pytest`.
