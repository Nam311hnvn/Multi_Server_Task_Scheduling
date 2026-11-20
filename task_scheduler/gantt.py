"""Dependency-free SVG Gantt chart generator for insertion in a course report."""
from __future__ import annotations
import argparse
from html import escape
import json
from pathlib import Path
from .model import TaskGraph
from .algorithms import schedule


def write_gantt(graph: TaskGraph, method: str, dest: Path) -> None:
    planned = schedule(graph, method)
    width = 1050
    left = 115
    right = 30
    top = 85
    lane = 64
    bottom = 80
    height = top + len(graph.machines) * lane + bottom
    graph_width = width - left - right
    scale = graph_width / planned.makespan
    colors = ['#196cb7', '#be7131', '#3d9177', '#8964a8', '#ad5252']
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="32" font-family="Arial" font-size="20" font-weight="bold" fill="#111">{escape(method.upper())} - Gantt chart</text>',
             f'<text x="{left}" y="52" font-family="Arial" font-size="12" fill="#555">Predicted makespan: {planned.makespan:.3f} time units</text>']
    for i in range(11):
        time = i * planned.makespan / 10
        x = left + graph_width * i / 10
        lines.append(f'<line x1="{x:.2f}" y1="{top-10}" x2="{x:.2f}" y2="{height-bottom+12}" stroke="#e4e7ea" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{height-bottom+37}" text-anchor="middle" font-size="12" font-family="Arial" fill="#777">{time:.1f}</text>')
    for idx, machine in enumerate(graph.machines):
        cy = top + idx * lane
        lines.append(f'<text x="{left-18}" y="{cy+27}" text-anchor="end" font-family="Arial" font-size="16" fill="#111">{escape(machine)}</text>')
        lines.append(f'<line x1="{left}" y1="{cy+48}" x2="{width-right}" y2="{cy+48}" stroke="#ddd"/>')
        tasks = sorted((p for p in planned.placements.values() if p.machine == machine), key=lambda p: p.start)
        for task in tasks:
            x = left + task.start * scale
            w = (task.finish - task.start) * scale
            fill = colors[idx % len(colors)]
            lines.append(f'<rect x="{x:.2f}" y="{cy+1}" width="{w:.2f}" height="42" rx="5" fill="{fill}"/>')
            if w > 22:
                lines.append(f'<text x="{x+w/2:.2f}" y="{cy+26}" text-anchor="middle" font-size="13" font-family="Arial" fill="white" font-weight="bold">{escape(task.task)}</text>')
    lines.append(f'<text x="{left + graph_width/2:.2f}" y="{height-13}" text-anchor="middle" font-family="Arial" font-size="14" fill="#444">Time (abstract scheduling units)</text>')
    lines.append('</svg>')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('\n'.join(lines), encoding='utf8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--algorithm', default='heft', choices=['heft','cpop','mxct','mnct','avbs'])
    p.add_argument('--output', type=Path, default=Path('outputs/gantt.svg'))
    args = p.parse_args()
    graph = TaskGraph.from_dict(json.loads(args.input.read_text(encoding='utf8')))
    write_gantt(graph, args.algorithm, args.output)
    print('Saved chart:', args.output)


if __name__ == '__main__':
    main()
