"""Paper metrics (makespan, SLR, speedup) + explicitly added load metrics."""
from __future__ import annotations
import statistics
from .algorithms import Schedule
from .model import TaskGraph


def evaluate(graph: TaskGraph, sched: Schedule) -> dict:
    makespan = sched.makespan
    busy = {m: sum(p.finish - p.start for p in sched.placements.values() if p.machine == m)
            for m in graph.machines}
    utilization = {m: busy[m] / makespan for m in graph.machines}
    vals = list(utilization.values())
    mean = statistics.mean(vals)
    cv = statistics.pstdev(vals) / mean if mean else 0.0
    sequential = min(sum(costs[i] for costs in graph.costs.values()) for i in range(len(graph.machines)))
    longest = {}
    pred = graph.predecessors()
    for t in graph.topological_order():
        longest[t] = min(graph.costs[t]) + max((longest[p] for p in pred[t]), default=0.0)
    lb = max(longest.values())  # communication-free critical path lower bound
    return {
        'makespan': makespan,
        'speedup': sequential / makespan,
        'slr': makespan / lb,
        'slr_lower_bound': lb,
        'busy_time': busy,
        'utilization': utilization,
        'load_cv': cv,
        'scheduler_ms': sched.scheduler_ms,
    }
