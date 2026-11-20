"""Reproducible random DAG generator for controlled comparisons.

This is a practical layered generator, not a byte-for-byte reproduction of
Topcuoglu's parametric graph generator.
"""
from __future__ import annotations
import random
from .model import TaskGraph


def random_graph(n: int = 20, m: int = 3, density: float = 0.2,
                 ccr: float = 1.0, seed: int = 42) -> TaskGraph:
    if n < 2 or m < 1 or not 0 <= density <= 1 or ccr < 0:
        raise ValueError('Invalid DAG generator parameters')
    rand = random.Random(seed)
    machines = [f'VM{i+1}' for i in range(m)]
    # Different task/machine runtimes, reproducible with seed
    durations = [[round(rand.uniform(4, 25) * rand.uniform(0.5, 1.5), 3)
                  for _ in range(m)] for _ in range(n)]
    avg = sum(sum(x) / m for x in durations) / n
    edges = []
    # A chain backbone ensures entry reaches exit; extra edges create parallel constraints
    for i in range(n - 1):
        if i % 4 == 0:
            edges.append((i, i+1))
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edges and rand.random() < density * (1.0 if j-i < 8 else 0.25):
                edges.append((i, j))
    # Single entry and exit, which also makes CPOP applicable
    incoming = {j for _, j in edges}
    outgoing = {i for i, _ in edges}
    for i in range(1, n):
        if i not in incoming:
            edges.append((0, i))
    for i in range(n - 1):
        if i not in outgoing:
            edges.append((i, n-1))
    edges = sorted(set(edges))
    return TaskGraph.from_dict({
        'machines': machines,
        'tasks': [{'id': f'T{i+1}', 'costs': x} for i, x in enumerate(durations)],
        'edges': [{'from': f'T{i+1}', 'to': f'T{j+1}',
                   'comm': round(ccr * avg * rand.uniform(0.5, 1.5), 3)} for i, j in edges],
    })
