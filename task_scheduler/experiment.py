"""Reproducible batch study, CSV table for chapter 4."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import statistics
from .generate import random_graph
from .algorithms import schedule
from .metrics import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', type=int, default=30)
    p.add_argument('--runs', type=int, default=30)
    p.add_argument('--machines', type=int, default=3)
    p.add_argument('--density', type=float, default=0.15)
    p.add_argument('--ccrs', default='0.1,1.0,5.0')
    p.add_argument('--seed-start', type=int, default=2000)
    p.add_argument('--output', type=Path, default=Path('outputs/batch.csv'))
    args = p.parse_args()
    if args.runs < 1:
        p.error('--runs must be >=1')
    rows = []
    for ccr in map(float, args.ccrs.split(',')):
        for seed in range(args.seed_start, args.seed_start + args.runs):
            graph = random_graph(n=args.tasks, m=args.machines, density=args.density, ccr=ccr, seed=seed)
            for method in ('heft','cpop','mxct','mnct','avbs'):
                scores = evaluate(graph, schedule(graph, method))
                rows.append({'seed': seed, 'tasks': args.tasks, 'machines': args.machines,
                             'ccr_parameter': ccr, 'algorithm': method,
                             'makespan': round(scores['makespan'], 4),
                             'slr': round(scores['slr'], 4),
                             'speedup': round(scores['speedup'], 4),
                             'load_cv': round(scores['load_cv'], 5),
                             'scheduler_ms': round(scores['scheduler_ms'], 5)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} observations to {args.output}')
    print(f"{'CCR':>6} {'Method':>10} {'Avg makespan':>16} {'Avg CV':>10}")
    for ccr in map(float, args.ccrs.split(',')):
        for method in ('heft','cpop','mxct','mnct','avbs'):
            group = [r for r in rows if r['ccr_parameter'] == ccr and r['algorithm'] == method]
            print(f'{ccr:>6.1f} {method:>10} {statistics.mean(r["makespan"] for r in group):>16.3f} {statistics.mean(r["load_cv"] for r in group):>10.3f}')


if __name__ == '__main__':
    main()
