"""List-scheduling HEFT/CPOP and 2023 rank-selection variants.

Key research distinction:
- AVCT = 2002 HEFT, averaging costs for upward rank.
- MXCT/MNCT = 2023 alternatives for rank calculation.
- AVBS is coded as exhaustive best-feasible-slot selection. In the standard
  fixed-duration, nonpreemptive model it is mathematically equivalent to
  selecting the earliest feasible slot. The 2023 AVBS result therefore needs
  extra methodological detail to reproduce; no artificial improvement is made.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
import statistics
from time import perf_counter
from .model import TaskGraph


@dataclass(frozen=True)
class Placement:
    task: str
    machine: str
    start: float
    finish: float

    def to_dict(self) -> dict:
        return {'task': self.task, 'machine': self.machine,
                'start': self.start, 'finish': self.finish}


@dataclass
class Schedule:
    method: str
    placements: dict[str, Placement]
    ranks: dict[str, float]
    scheduler_ms: float
    critical_path: list[str] | None = None
    cp_machine: str | None = None

    @property
    def makespan(self) -> float:
        return max((p.finish for p in self.placements.values()), default=0.0)

    def to_dict(self) -> dict:
        return {'method': self.method, 'makespan': self.makespan,
                'scheduler_ms': self.scheduler_ms,
                'ranks': self.ranks,
                'placements': [p.to_dict() for p in sorted(self.placements.values(), key=lambda p: (p.start, p.task))],
                'critical_path': self.critical_path, 'cp_machine': self.cp_machine}


def task_weight(costs: tuple[float, ...], mode: str) -> float:
    if mode == 'mean':
        return statistics.mean(costs)
    if mode == 'max':
        return max(costs)
    if mode == 'min':
        return min(costs)
    raise ValueError('Unknown weight mode')


def upward_ranks(graph: TaskGraph, mode: str = 'mean') -> dict[str, float]:
    succ = graph.successors()
    ranks = {}
    for t in reversed(graph.topological_order()):
        downstream = max((graph.edges[t, s] + ranks[s] for s in succ[t]), default=0.0)
        ranks[t] = task_weight(graph.costs[t], mode) + downstream
    return ranks


def downward_ranks(graph: TaskGraph) -> dict[str, float]:
    pred = graph.predecessors()
    ranks = {}
    for t in graph.topological_order():
        ranks[t] = max((ranks[p] + statistics.mean(graph.costs[p]) + graph.edges[p, t]
                        for p in pred[t]), default=0.0)
    return ranks


def feasible_start(occupied: list[Placement], ready: float, duration: float, scan_all: bool = False) -> float:
    """Return earliest feasible gap in sorted processor calendar.

    scan_all=True enumerates gaps explicitly; each feasible gap has completion
    time start + fixed duration, so earliest feasible start is always best.
    """
    occupied = sorted(occupied, key=lambda p: p.start)
    cursor = max(0.0, ready)
    choices = []
    for item in occupied:
        if cursor + duration <= item.start + 1e-9:
            choices.append(cursor)
            if not scan_all:
                return cursor
        cursor = max(cursor, item.finish)
    choices.append(cursor)
    return min(choices)  # earliest feasible slot also minimizes finish time


def schedule(graph: TaskGraph, method: str = 'heft') -> Schedule:
    method = method.lower()
    if method not in ('heft', 'avct', 'mxct', 'mnct', 'avbs', 'cpop'):
        raise ValueError('method must be heft, avct, mxct, mnct, avbs, or cpop')
    start_clock = perf_counter()
    mode = {'mxct': 'max', 'mnct': 'min'}.get(method, 'mean')
    up = upward_ranks(graph, mode)
    priorities = up.copy()
    cp: list[str] | None = None
    cp_machine: str | None = None
    if method == 'cpop':
        down = downward_ranks(graph)
        priorities = {t: up[t] + down[t] for t in graph.costs}
        entry = [t for t in graph.costs if not graph.predecessors()[t]]
        if len(entry) != 1:
            raise ValueError('CPOP demonstration requires exactly one entry; use zero-cost pseudo-entry for multi-entry DAGs')
        cp = [entry[0]]
        succ = graph.successors()
        target = priorities[entry[0]]
        while succ[cp[-1]]:
            options = [x for x in succ[cp[-1]] if math.isclose(priorities[x], target, rel_tol=1e-9, abs_tol=1e-8)]
            if not options:
                raise ValueError('Critical-path successor not found; check graph data')
            cp.append(options[0])   # first in edge order (2002 tie rule)
        best_j = min(range(len(graph.machines)), key=lambda j: sum(graph.costs[t][j] for t in cp))
        cp_machine = graph.machines[best_j]
    predecessor = graph.predecessors()
    successor = graph.successors()
    degree = {t: len(predecessor[t]) for t in graph.costs}
    order_index = {t: i for i, t in enumerate(graph.costs)}
    ready = [(-priorities[t], order_index[t], t) for t in graph.costs if degree[t] == 0]
    heapq.heapify(ready)
    placed = {}
    calendars = {m: [] for m in graph.machines}
    while ready:
        _, _, task = heapq.heappop(ready)
        candidates = []
        for j, machine in enumerate(graph.machines):
            if method == 'cpop' and task in cp and machine != cp_machine:
                continue
            arrivals = [placed[p].finish + (0 if placed[p].machine == machine else graph.edges[p, task])
                        for p in predecessor[task]]
            task_ready = max(arrivals, default=0.0)
            duration = graph.costs[task][j]
            t_start = feasible_start(calendars[machine], task_ready, duration, method == 'avbs')
            candidates.append((t_start + duration, j, t_start))
        finish, machine_j, t_start = min(candidates)
        choice = Placement(task, graph.machines[machine_j], t_start, finish)
        placed[task] = choice
        calendars[choice.machine].append(choice)
        for next_task in successor[task]:
            degree[next_task] -= 1
            if degree[next_task] == 0:
                heapq.heappush(ready, (-priorities[next_task], order_index[next_task], next_task))
    answer = Schedule(method, placed, priorities, 1000 * (perf_counter() - start_clock), cp, cp_machine)
    validate_schedule(graph, answer)
    return answer


def validate_schedule(graph: TaskGraph, result: Schedule) -> None:
    if set(result.placements) != set(graph.costs):
        raise AssertionError('Tasks missing from schedule')
    for t, p in result.placements.items():
        index = graph.machines.index(p.machine)
        if abs(p.finish - p.start - graph.costs[t][index]) > 1e-6:
            raise AssertionError(f'Incorrect duration for task {t}')
        if p.start < -1e-9:
            raise AssertionError('Negative start time')
    for machine in graph.machines:
        intervals = sorted((x for x in result.placements.values() if x.machine == machine), key=lambda x: x.start)
        for a, b in zip(intervals, intervals[1:]):
            if a.finish > b.start + 1e-7:
                raise AssertionError(f'Overlapping tasks on {machine}')
    for (u, v), comm in graph.edges.items():
        first, second = result.placements[u], result.placements[v]
        cost = 0 if first.machine == second.machine else comm
        if second.start + 1e-7 < first.finish + cost:
            raise AssertionError(f'Precedence violation {u}->{v}')
