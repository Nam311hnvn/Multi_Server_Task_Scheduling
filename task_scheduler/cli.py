"""CLI: simulate a schedule, validate it, export JSON for the report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .model import TaskGraph
from .algorithms import schedule
from .metrics import evaluate
from .generate import random_graph


def run_cli() -> None:
    p = argparse.ArgumentParser(description='HEFT / CPOP / MXCT / MNCT / AVBS research lab')
    p.add_argument('--input', type=Path, help='DAG JSON input')
    p.add_argument('--generate', type=int, metavar='TASKS', help='Generate random DAG')
    p.add_argument('--ccr', type=float, default=1.0)
    p.add_argument('--density', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--machines', type=int, default=3)
    p.add_argument('--method', default='all', choices=['all','heft','avct','mxct','mnct','avbs','cpop'])
    p.add_argument('--output', type=Path, help='Save results to JSON')
    args = p.parse_args()
    if args.input and args.generate:
        p.error('Choose --input or --generate, not both')
    if args.input:
        graph = TaskGraph.from_dict(json.loads(args.input.read_text(encoding='utf8')))
    elif args.generate:
        graph = random_graph(args.generate, args.machines, args.density, args.ccr, args.seed)
    else:
        p.error('Provide --input or --generate')
    methods = ['heft','cpop','mxct','mnct','avbs'] if args.method == 'all' else [args.method]
    results = []
    print(f"Tasks={len(graph.costs)}  Edges={len(graph.edges)}  Machines={len(graph.machines)}")
    print(f"{'Algorithm':<12}{'Makespan':>12}{'SLR':>10}{'Speedup':>12}{'Load CV':>11}{'Scheduler ms':>16}")
    for method in methods:
        plan = schedule(graph, method)
        metrics = evaluate(graph, plan)
        results.append({'schedule': plan.to_dict(), 'metrics': metrics})
        print(f"{method.upper():<12}{metrics['makespan']:>12.3f}{metrics['slr']:>10.3f}{metrics['speedup']:>12.3f}{metrics['load_cv']:>11.3f}{metrics['scheduler_ms']:>16.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'input': graph.to_dict(), 'results': results}, ensure_ascii=False, indent=2), encoding='utf8')
        print('Saved', args.output)


if __name__ == '__main__':
    run_cli()
