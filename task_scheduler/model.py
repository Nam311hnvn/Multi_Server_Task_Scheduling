"""Validated workload model. Units: arbitrary time units (or milliseconds if chosen).

Each edge's comm is its inter-VM communication delay in the same units. The
intra-VM communication delay is zero. This is a simplified instance of the
Topcuoglu et al. (2002) communication model.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any
import math


def finite_number(x: Any, name: str, positive: bool = False) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite numeric value")
    if x < 0 or (positive and x <= 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(x)


@dataclass(frozen=True)
class TaskGraph:
    machines: tuple[str, ...]
    costs: dict[str, tuple[float, ...]]
    edges: dict[tuple[str, str], float]

    @classmethod
    def from_dict(cls, data: dict) -> 'TaskGraph':
        if not isinstance(data, dict):
            raise ValueError('Input must be a JSON object')
        machines = tuple(data['machines'])
        if not machines or len(set(machines)) != len(machines) or any(not isinstance(m, str) or not m for m in machines):
            raise ValueError('machines must be nonempty and unique strings')
        tasks = data['tasks']
        if not isinstance(tasks, list) or not tasks:
            raise ValueError('tasks must be a nonempty list')
        costs = {}
        for task in tasks:
            name = task['id']
            if not isinstance(name, str) or not name or name in costs:
                raise ValueError('Duplicate or invalid task id')
            values = task['costs']
            if len(values) != len(machines):
                raise ValueError(f'Wrong number of costs for {name}')
            costs[name] = tuple(finite_number(x, f'costs[{name}]', positive=True) for x in values)
        edges = {}
        for edge in data.get('edges', []):
            u, v = edge['from'], edge['to']
            if u not in costs or v not in costs or u == v or (u, v) in edges:
                raise ValueError('Invalid, repeated or self-loop edge')
            edges[(u, v)] = finite_number(edge['comm'], f'edge {u}->{v}')
        graph = cls(machines, costs, edges)
        graph.topological_order()   # Raises if cyclic
        return graph

    def to_dict(self) -> dict:
        return {
            'machines': list(self.machines),
            'tasks': [{'id': t, 'costs': list(v)} for t, v in self.costs.items()],
            'edges': [{'from': a, 'to': b, 'comm': c} for (a, b), c in self.edges.items()],
        }

    def predecessors(self) -> dict[str, list[str]]:
        p = {x: [] for x in self.costs}
        for u, v in self.edges:
            p[v].append(u)
        return p

    def successors(self) -> dict[str, list[str]]:
        s = {x: [] for x in self.costs}
        for u, v in self.edges:
            s[u].append(v)
        return s

    def topological_order(self) -> list[str]:
        preds = self.predecessors()
        succs = self.successors()
        degree = {x: len(preds[x]) for x in self.costs}
        queue = deque(x for x in self.costs if degree[x] == 0)
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in succs[u]:
                degree[v] -= 1
                if degree[v] == 0:
                    queue.append(v)
        if len(order) != len(self.costs):
            raise ValueError('DAG has a directed cycle')
        return order
