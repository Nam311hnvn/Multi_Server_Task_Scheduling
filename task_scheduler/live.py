"""Replay a static schedule across 1..N HTTP workers.

These remote workers emulate computation by sleep. Results measure orchestration
and synchronization, not CPU throughput or physical network data-transfer time.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen
from .model import TaskGraph
from .algorithms import schedule
from .metrics import evaluate


def request_json(url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    body = None if payload is None else json.dumps(payload).encode('utf8')
    req = Request(
        url,
        data=body,
        method='GET' if body is None else 'POST',
        headers={'Content-Type': 'application/json'}
    )
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode('utf8'))


def run_live(graph: TaskGraph, algorithm: str, endpoints: list[str],
             time_unit_ms: float = 20.0) -> dict:
    if len(endpoints) != len(graph.machines):
        raise ValueError('Exactly one endpoint must be given for each machine, in graph order')
    if not 0 < time_unit_ms <= 1000:
        raise ValueError('time_unit_ms must be in (0, 1000]')
    plan = schedule(graph, algorithm)
    if len(set(endpoints)) != len(endpoints):
        raise ValueError('Worker endpoints must be unique; one endpoint per machine')
    # Verify worker identity rather than assuming every endpoint resolves to a different VM.
    worker_nodes = []
    for url in endpoints:
        if not url.startswith('http://'):
            raise ValueError('This lab supports http:// on trusted PRIVATE network only')
        health = request_json(url.rstrip('/') + '/health', timeout=5)
        if not health.get('ok') or not isinstance(health.get('node'), str) or not health['node'].strip():
            raise ValueError(f'Worker health check or node identity failed for {url}')
        worker_nodes.append(health['node'])
    if len(set(worker_nodes)) != len(worker_nodes):
        raise ValueError(f'Worker node identities must be unique: {worker_nodes}')
    machine_map = dict(zip(graph.machines, endpoints))
    worker_mapping = {machine: {'node': node, 'endpoint': endpoint}
                      for machine, node, endpoint in zip(graph.machines, worker_nodes, endpoints)}
    machine_schedule = {m: sorted((p for p in plan.placements.values() if p.machine == m),
                                  key=lambda p: p.start) for m in graph.machines}
    events = {t: threading.Event() for t in graph.costs}
    measured_finish = {}
    errors = {}
    records = {}
    predecessors = graph.predecessors()
    t0 = time.perf_counter()
    unit_s = time_unit_ms / 1000

    def serve_machine(machine: str):
        for placement in machine_schedule[machine]:
            task = placement.task
            try:
                # Wait for ALL dependencies, including those on other VMs.
                for parent in predecessors[task]:
                    events[parent].wait()
                    if parent in errors:
                        raise RuntimeError(f'Dependency {parent} failed')
                planned_start = t0 + placement.start * unit_s
                ready = max((measured_finish[parent] +
                             (0 if plan.placements[parent].machine == machine
                              else graph.edges[parent, task] * unit_s)
                             for parent in predecessors[task]), default=t0)
                earliest = max(planned_start, ready)
                remaining = earliest - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                actual_start = time.perf_counter()
                result = request_json(machine_map[machine].rstrip('/') + '/execute',
                      {'task': task, 'duration_ms':
                       (placement.finish - placement.start) * time_unit_ms},
                      timeout=max(30, (placement.finish-placement.start)*unit_s+20))
                if result.get('task') != task:
                    raise RuntimeError('Worker returned a different task id')
                if result.get('node') != worker_mapping[machine]['node']:
                    raise RuntimeError(f'Worker identity changed while executing {task}')
                actual_finish = time.perf_counter()
                measured_finish[task] = actual_finish
                records[task] = {'task': task, 'machine': machine,
                                 'predicted_start_unit': placement.start,
                                 'predicted_finish_unit': placement.finish,
                                 'actual_start_ms': (actual_start-t0)*1000,
                                 'actual_finish_ms': (actual_finish-t0)*1000,
                                 'worker_reported_ms': result['elapsed_ms']}
            except Exception as exc:
                errors[task] = repr(exc)
            finally:
                events[task].set()
    with ThreadPoolExecutor(max_workers=len(graph.machines)) as pool:
        futures = [pool.submit(serve_machine, m) for m in graph.machines]
        for future in futures:
            future.result()
    if errors:
        raise RuntimeError(f'Live execution failed: {errors}')
    ms = max(x['actual_finish_ms'] for x in records.values())
    return {'algorithm': algorithm,
            'predicted': evaluate(graph, plan),
            'worker_mapping': worker_mapping,
            'actual_makespan_ms': ms,
            'time_unit_ms': time_unit_ms,
            'note': 'Synthetic sleep-based tasks and simulated inter-VM comm delays; no payload transfer or CPU benchmark.',
            'tasks': sorted(records.values(), key=lambda x: x['actual_start_ms'])}


def main():
    parser = argparse.ArgumentParser(description='Execute simulated DAG via remote HTTP workers')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--endpoints', required=True, help='Comma-separated HTTP URLs in machines order')
    parser.add_argument('--algorithm', default='heft', choices=['heft','cpop','mxct','mnct','avbs'])
    parser.add_argument('--time-unit-ms', type=float, default=20)
    parser.add_argument('--output', type=Path, default=Path('outputs/live.json'))
    args = parser.parse_args()
    graph = TaskGraph.from_dict(json.loads(args.input.read_text(encoding='utf8')))
    result = run_live(graph, args.algorithm, args.endpoints.split(','), token, args.time_unit_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    print(f"{args.algorithm.upper()} predicted makespan = {result['predicted']['makespan']:.3f} time units")
    print(f"Actual wall makespan = {result['actual_makespan_ms']:.1f} ms")
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
