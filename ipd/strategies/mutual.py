"""Desperate, Hopeless and Willing."""
from ipd.bots import memory_one

# each is memory-one with a random (p = 0.5) first move
STRATEGIES = [
    memory_one([0, 0, 0, 1], p0=0.5, name="Desperate"),
    memory_one([0, 1, 1, 1], p0=0.5, name="Hopeless"),
    memory_one([1, 1, 1, 0], p0=0.5, name="Willing"),
]
