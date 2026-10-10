"""Every ported axelrod library strategy, keyed by library name.

Each module ports the axelrod module of the same name (axelrod_second is split in two). The ports follow
the library's code where it differs from the published descriptions.
"""
import importlib
import pkgutil

from ipd import bots


def all_strategies():
    """All ported bots (classics included), in a stable order. Imports every module on first call."""
    found = list(bots.CLASSICS) + [bots.generous_tit_for_tat(), bots.random_bot()]
    for m in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        found += importlib.import_module(f"{__name__}.{m.name}").STRATEGIES
    return found


def by_name():
    return {b.name: b for b in all_strategies()}
