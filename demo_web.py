#!/usr/bin/env python3
"""
Interactive Web Demo: 3-VM Task Scheduling Simulator (HEFT vs CPOP & Variants)
Standard-library-only Python 3.10+ (khong can pip install).
Chay lenh: python demo_web.py  (mac dinh cong 5050)
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlparse


# =============================================================================
# 1. DU LIEU CHUAN TOPCUOGLU ET AL. (2002) - HINH 3 (10 TAC VU, 15 CANH, 3 MAY)
# =============================================================================
def get_paper2002_dag() -> dict:
    tasks = [f"n{i}" for i in range(1, 11)]
    processors = ["P1", "P2", "P3"]
    W = {
        "n1": [14, 16, 9],
        "n2": [13, 19, 18],
        "n3": [11, 13, 19],
        "n4": [13, 8, 17],
        "n5": [12, 13, 10],
        "n6": [13, 16, 9],
        "n7": [7, 15, 11],
        "n8": [5, 11, 14],
        "n9": [18, 12, 20],
        "n10": [21, 7, 16],
    }
    edges = [
        ("n1", "n2", 18), ("n1", "n3", 12), ("n1", "n4", 9),
        ("n1", "n5", 11), ("n1", "n6", 14),
        ("n2", "n8", 19), ("n2", "n9", 16),
        ("n3", "n7", 23),
        ("n4", "n8", 27), ("n4", "n9", 23),
        ("n5", "n9", 13),
        ("n6", "n8", 15),
        ("n7", "n10", 17), ("n8", "n10", 11), ("n9", "n10", 13),
    ]
    return {
        "name": "Topcuoglu et al. (2002) — Fig. 3 (10 tác vụ, 15 cạnh)",
        "tasks": tasks,
        "processors": processors,
        "W": W,
        "edges": [{"src": u, "dst": v, "comm": c} for u, v, c in edges],
    }


def generate_random_dag(num_tasks: int = 15, ccr: float = 1.0, density: float = 0.15, seed: int = 2026) -> dict:
    rng = random.Random(seed)
    num_tasks = max(6, min(100, num_tasks))
    density = max(0.05, min(0.6, density))
    tasks = [f"n{i}" for i in range(1, num_tasks + 1)]
    processors = ["P1", "P2", "P3"]

    W: Dict[str, List[int]] = {}
    all_w = []
    for t in tasks:
        base = rng.randint(8, 24)
        costs = [
            max(3, int(round(base * rng.uniform(0.65, 1.35)))),
            max(3, int(round(base * rng.uniform(0.65, 1.35)))),
            max(3, int(round(base * rng.uniform(0.65, 1.35)))),
        ]
        W[t] = costs
        all_w.extend(costs)

    avg_w = sum(all_w) / len(all_w)
    avg_c = max(1.0, avg_w * ccr)

    adj = set()
    for j in range(1, num_tasks):
        p = rng.randint(max(0, j - 4), j - 1)
        adj.add((p, j))
    for i in range(0, num_tasks - 1):
        if not any(u == i for u, _ in adj):
            v = rng.randint(i + 1, min(num_tasks - 1, i + 4))
            adj.add((i, v))
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if (i, j) not in adj and rng.random() < density:
                adj.add((i, j))

    edges = []
    for i, j in sorted(adj):
        c = max(1, int(round(rng.uniform(0.5 * avg_c, 1.5 * avg_c))))
        edges.append({"src": tasks[i], "dst": tasks[j], "comm": c})

    return {
        "name": f"DAG ngẫu nhiên ({num_tasks} tác vụ, {len(edges)} cạnh, CCR={ccr}, density={density}, seed={seed})",
        "tasks": tasks,
        "processors": processors,
        "W": W,
        "edges": edges,
    }


# =============================================================================
# 2. LOI THUAT TOAN: HEFT, CPOP, MXCT, MNCT, AVBS* & CHI SO DANH GIA
# =============================================================================
def build_graph_maps(dag: dict):
    tasks = dag["tasks"]
    succ: Dict[str, List[Tuple[str, float]]] = {t: [] for t in tasks}
    pred: Dict[str, List[Tuple[str, float]]] = {t: [] for t in tasks}
    for e in dag["edges"]:
        u, v, c = e["src"], e["dst"], float(e["comm"])
        succ[u].append((v, c))
        pred[v].append((u, c))

    indeg = {t: len(pred[t]) for t in tasks}
    q = [t for t in tasks if indeg[t] == 0]
    topo: List[str] = []
    while q:
        u = q.pop(0)
        topo.append(u)
        for v, _ in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return succ, pred, topo


def compute_upward_ranks(dag: dict, succ, topo, mode: str = "mean") -> Dict[str, float]:
    W = dag["W"]
    rank_u: Dict[str, float] = {}
    for t in reversed(topo):
        if mode == "mean":
            w_val = sum(W[t]) / len(W[t])
        elif mode == "max":
            w_val = float(max(W[t]))
        elif mode == "min":
            w_val = float(min(W[t]))
        else:
            raise ValueError(mode)
        if not succ[t]:
            rank_u[t] = w_val
        else:
            rank_u[t] = w_val + max(c + rank_u[v] for v, c in succ[t])
    return rank_u


def compute_downward_ranks(dag: dict, pred, topo) -> Dict[str, float]:
    W = dag["W"]
    rank_d: Dict[str, float] = {}
    for t in topo:
        if not pred[t]:
            rank_d[t] = 0.0
        else:
            rank_d[t] = max(
                rank_d[u] + (sum(W[u]) / len(W[u])) + c
                for u, c in pred[t]
            )
    return rank_d


def feasible_start_first_fit(intervals: List[Tuple[float, float, str]], ready: float, duration: float) -> float:
    x = ready
    for s, f, _ in intervals:
        if x + duration <= s + 1e-9:
            return x
        if f > x:
            x = f
    return x


def feasible_start_avbs(intervals: List[Tuple[float, float, str]], ready: float, duration: float) -> float:
    best_eft = float("inf")
    best_start = ready
    prev_end = 0.0
    for s, f, _ in intervals:
        cand = max(ready, prev_end)
        if cand + duration <= s + 1e-9:
            eft = cand + duration
            if eft < best_eft - 1e-9:
                best_eft = eft
                best_start = cand
        prev_end = max(prev_end, f)
    cand_end = max(ready, prev_end)
    if cand_end + duration < best_eft - 1e-9:
        best_start = cand_end
    return best_start


def ready_time(task: str, proc_idx: int, pred, assigned: Dict[str, Tuple[int, float, float]]) -> float:
    if not pred[task]:
        return 0.0
    best = 0.0
    for u, comm in pred[task]:
        u_proc, _, u_aft = assigned[u]
        arrival = u_aft + (0.0 if u_proc == proc_idx else comm)
        if arrival > best:
            best = arrival
    return best


def compute_metrics(dag: dict, pred, topo, placements: List[dict], scheduler_ms: float) -> dict:
    W = dag["W"]
    procs = dag["processors"]
    q = len(procs)
    makespan = max((p["finish"] for p in placements), default=0.0)

    dp_lb: Dict[str, float] = {}
    for t in topo:
        min_w = float(min(W[t]))
        dp_lb[t] = min_w + (max(dp_lb[u] for u, _ in pred[t]) if pred[t] else 0.0)
    lb = max(dp_lb.values()) if dp_lb else 1.0

    best_seq = min(sum(W[t][j] for t in dag["tasks"]) for j in range(q))

    busy = [0.0] * q
    for pl in placements:
        busy[pl["proc_idx"]] += pl["duration"]
    util = [b / makespan if makespan > 0 else 0.0 for b in busy]
    mean_u = sum(util) / q if q > 0 else 0.0
    std_u = math.sqrt(sum((u - mean_u) ** 2 for u in util) / q) if q > 0 else 0.0
    cv_load = std_u / mean_u if mean_u > 0 else 0.0

    return {
        "makespan": round(makespan, 4),
        "lb": round(lb, 4),
        "best_seq": round(best_seq, 4),
        "slr": round(makespan / lb, 4) if lb > 0 else 0.0,
        "speedup": round(best_seq / makespan, 4) if makespan > 0 else 0.0,
        "busy_time": [round(b, 4) for b in busy],
        "utilization": [round(u * 100.0, 2) for u in util],
        "cv_load": round(cv_load, 4),
        "scheduler_ms": round(scheduler_ms, 4),
    }


def run_list_scheduler(dag: dict, method: str) -> dict:
    t0 = time.perf_counter()
    succ, pred, topo = build_graph_maps(dag)
    tasks = dag["tasks"]
    procs = dag["processors"]
    W = dag["W"]
    idx_map = {t: i for i, t in enumerate(tasks)}

    intervals: List[List[Tuple[float, float, str]]] = [[] for _ in procs]
    assigned: Dict[str, Tuple[int, float, float]] = {}
    placements: List[dict] = []
    critical_path: List[str] = []
    cp_proc_idx: int | None = None
    rank_d: Dict[str, float] = {}

    if method == "cpop":
        rank_u = compute_upward_ranks(dag, succ, topo, mode="mean")
        rank_d = compute_downward_ranks(dag, pred, topo)
        priority = {t: round(rank_u[t] + rank_d[t], 9) for t in tasks}

        entries = [t for t in tasks if not pred[t]]
        curr = max(entries, key=lambda t: (priority[t], -idx_map[t]))
        cp_len = priority[curr]
        critical_path = [curr]
        while succ[curr]:
            nxt = max(
                [v for v, _ in succ[curr]],
                key=lambda v: (1 if abs(priority[v] - cp_len) < 1e-6 else 0, priority[v], -idx_map[v]),
            )
            critical_path.append(nxt)
            curr = nxt
        cp_set = set(critical_path)
        cp_proc_idx = min(range(len(procs)), key=lambda j: (sum(W[t][j] for t in critical_path), j))

        remaining_preds = {t: len(pred[t]) for t in tasks}
        heap: List[Tuple[float, int, str]] = []
        for t in entries:
            heapq.heappush(heap, (-priority[t], idx_map[t], t))

        step = 0
        while heap:
            _, _, task = heapq.heappop(heap)
            step += 1
            if task in cp_set:
                j = cp_proc_idx
                r = ready_time(task, j, pred, assigned)
                dur = float(W[task][j])
                s = feasible_start_first_fit(intervals[j], r, dur)
                f = s + dur
                chosen_j, chosen_s, chosen_f, chosen_r = j, s, f, r
                reason = f"Thuộc đường găng → ép cố định lên {procs[j]}"
            else:
                best = None
                for j in range(len(procs)):
                    r = ready_time(task, j, pred, assigned)
                    dur = float(W[task][j])
                    s = feasible_start_first_fit(intervals[j], r, dur)
                    f = s + dur
                    cand = (f, j, s, r)
                    if best is None or cand < best:
                        best = cand
                chosen_f, chosen_j, chosen_s, chosen_r = best
                reason = f"Ngoài đường găng → chọn EFT nhỏ nhất ({procs[chosen_j]}: {chosen_f:.0f})"

            intervals[chosen_j].append((chosen_s, chosen_f, task))
            intervals[chosen_j].sort(key=lambda x: x[0])
            assigned[task] = (chosen_j, chosen_s, chosen_f)
            placements.append({
                "step": step,
                "task": task,
                "proc": procs[chosen_j],
                "proc_idx": chosen_j,
                "ready": round(chosen_r, 3),
                "start": round(chosen_s, 3),
                "finish": round(chosen_f, 3),
                "duration": float(W[task][chosen_j]),
                "is_cp": task in cp_set,
                "reason": reason,
            })
            for v, _ in succ[task]:
                remaining_preds[v] -= 1
                if remaining_preds[v] == 0:
                    heapq.heappush(heap, (-priority[v], idx_map[v], v))
    else:
        mode = {"heft": "mean", "avbs": "mean", "mxct": "max", "mnct": "min"}[method]
        rank_u = compute_upward_ranks(dag, succ, topo, mode=mode)
        # Lam tron 9 chu so thap phan de tranh sai so dau phay dong (n3 dung truoc n4 chuan IEEE 2002)
        order = sorted(tasks, key=lambda t: (-round(rank_u[t], 9), idx_map[t]))
        slot_fn = feasible_start_avbs if method == "avbs" else feasible_start_first_fit

        for step, task in enumerate(order, start=1):
            best = None
            eft_notes = []
            for j in range(len(procs)):
                r = ready_time(task, j, pred, assigned)
                dur = float(W[task][j])
                s = slot_fn(intervals[j], r, dur)
                f = s + dur
                eft_notes.append(f"{procs[j]}:{f:.0f}")
                cand = (f, j, s, r)
                if best is None or cand < best:
                    best = cand
            chosen_f, chosen_j, chosen_s, chosen_r = best
            intervals[chosen_j].append((chosen_s, chosen_f, task))
            intervals[chosen_j].sort(key=lambda x: x[0])
            assigned[task] = (chosen_j, chosen_s, chosen_f)
            placements.append({
                "step": step,
                "task": task,
                "proc": procs[chosen_j],
                "proc_idx": chosen_j,
                "ready": round(chosen_r, 3),
                "start": round(chosen_s, 3),
                "finish": round(chosen_f, 3),
                "duration": float(W[task][chosen_j]),
                "is_cp": False,
                "reason": f"EFT ({', '.join(eft_notes)}) → chọn {procs[chosen_j]}",
            })

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    metrics = compute_metrics(dag, pred, topo, placements, elapsed_ms)

    return {
        "method": method,
        "metrics": metrics,
        "placements": placements,
        "rank_u": {k: round(v, 3) for k, v in rank_u.items()},
        "rank_d": {k: round(v, 3) for k, v in rank_d.items()},
        "critical_path": critical_path,
        "cp_processor": procs[cp_proc_idx] if cp_proc_idx is not None else None,
    }


def evaluate_all(dag: dict) -> dict:
    methods = ["heft", "cpop", "mxct", "mnct", "avbs"]
    results = {m: run_list_scheduler(dag, m) for m in methods}
    succ, pred, topo = build_graph_maps(dag)
    levels: Dict[str, int] = {}
    for t in topo:
        levels[t] = 0 if not pred[t] else (max(levels[u] for u, _ in pred[t]) + 1)
    return {"dag": dag, "levels": levels, "results": results}


# =============================================================================
# 3. GIAO DIEN WEB TRUC QUAN (HTML + CSS + JS) — CAN DOI 50/50 MOI QUY MO
# =============================================================================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Demo 3 VM: So sánh HEFT vs CPOP — Thuật toán Nâng cao</title>
<style>
  :root {
    --bg: #f4f7fb;
    --card: #ffffff;
    --border: #dce3ee;
    --text: #1e293b;
    --muted: #64748b;
    --p1: #1d4ed8;
    --p2: #c2410c;
    --p3: #0f766e;
    --accent: #0f172a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.45;
  }
  header {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    color: #fff; padding: 14px 24px;
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;
  }
  header h1 { margin: 0; font-size: 1.2rem; letter-spacing: 0.2px; }
  header p { margin: 2px 0 0; font-size: 0.84rem; color: #cbd5e1; }
  .badge-header {
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.22);
    padding: 6px 12px; border-radius: 999px; font-size: 0.78rem; white-space: nowrap;
  }
  .container { max-width: 1440px; margin: 14px auto; padding: 0 16px 32px; }
  .panel {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px 16px; margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    min-width: 0; /* Khoa chong tran luoi CSS Grid */
  }
  .controls-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
  .controls-row label { font-size: 0.84rem; color: var(--muted); font-weight: 600; }
  select, input[type="number"] {
    padding: 6px 10px; border: 1px solid var(--border); border-radius: 8px;
    font-size: 0.86rem; background: #fff; color: var(--text);
  }
  button {
    padding: 7px 13px; border-radius: 8px; border: 1px solid var(--border);
    background: #fff; color: var(--text); font-weight: 600; font-size: 0.86rem;
    cursor: pointer; transition: 0.15s; white-space: nowrap;
  }
  button:hover { background: #f1f5f9; }
  button.primary { background: #1d4ed8; color: #fff; border-color: #1d4ed8; }
  button.primary:hover { background: #1e40af; }
  button.success { background: #0f766e; color: #fff; border-color: #0f766e; }
  button.success:hover { background: #115e59; }
  .playback-bar {
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    background: #f8fafc; border: 1px solid var(--border);
    padding: 10px 14px; border-radius: 10px; margin-top: 10px;
  }
  .clock-box {
    font-family: Consolas, monospace; font-size: 1rem; font-weight: 700;
    background: #0f172a; color: #38bdf8; padding: 6px 12px; border-radius: 8px;
    min-width: 158px; text-align: center; white-space: nowrap;
  }
  input[type="range"] { flex: 1; min-width: 160px; accent-color: #1d4ed8; }

  /* KHOA CAN DOI TUYET DOI 50/50 CHO 2 COT HEFT & CPOP */
  .race-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 14px;
    align-items: start;
  }
  @media (max-width: 1024px) { .race-grid { grid-template-columns: minmax(0, 1fr); } }

  .algo-header {
    display: flex; justify-content: space-between; align-items: center; gap: 10px;
    margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border);
    min-height: 44px;
  }
  .algo-title {
    display: flex; align-items: center; gap: 8px; min-width: 0; flex: 1;
  }
  .algo-subtitle {
    font-size: 0.92rem; font-weight: 700; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .pill {
    font-size: 0.76rem; padding: 4px 10px; border-radius: 999px; font-weight: 700;
    white-space: nowrap; flex-shrink: 0;
  }
  .pill-heft { background: #dbeafe; color: #1e40af; }
  .pill-cpop { background: #ffedd5; color: #9a3412; }
  .pill-done { background: #dcfce7; color: #166534; }
  .pill-run { background: #fef3c7; color: #92400e; }

  .vm-grid {
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px; margin-bottom: 12px;
  }
  .vm-card {
    border: 1.5px solid var(--border); border-radius: 10px; padding: 9px 10px;
    background: #f8fafc; transition: all 0.15s; min-width: 0;
  }
  .vm-card.active-p1 { border-color: var(--p1); background: #eff6ff; box-shadow: 0 0 0 2px rgba(29,78,216,0.12); }
  .vm-card.active-p2 { border-color: var(--p2); background: #fff7ed; box-shadow: 0 0 0 2px rgba(194,65,12,0.12); }
  .vm-card.active-p3 { border-color: var(--p3); background: #f0fdfa; box-shadow: 0 0 0 2px rgba(15,118,110,0.12); }
  .vm-top {
    display: flex; justify-content: space-between; align-items: center; gap: 4px;
    font-size: 0.76rem; font-weight: 700; white-space: nowrap; overflow: hidden;
  }
  .vm-spec { color: #64748b; font-weight: 500; font-size: 0.72rem; }
  .vm-ip { font-size: 0.71rem; color: var(--muted); font-family: Consolas, monospace; margin-top: 2px; }
  .vm-state {
    margin-top: 6px; font-size: 0.78rem; font-weight: 700; padding: 4px 6px;
    border-radius: 6px; text-align: center; background: #e2e8f0; color: #475569;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .vm-state.running { background: #0f172a; color: #fff; }
  .vm-bar-wrap { height: 6px; background: #e2e8f0; border-radius: 999px; margin-top: 6px; overflow: hidden; }
  .vm-bar { height: 100%; width: 0%; transition: width 0.08s linear; }

  .gantt-wrap { width: 100%; overflow-x: auto; }
  .metrics-strip {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px; margin-top: 10px;
  }
  .metric-box {
    background: #f8fafc; border: 1px solid var(--border); border-radius: 8px;
    padding: 7px 6px; text-align: center; min-width: 0;
  }
  .metric-box .lbl { font-size: 0.72rem; color: var(--muted); white-space: nowrap; }
  .metric-box .val { font-size: 0.98rem; font-weight: 700; color: var(--accent); }

  .bottom-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr);
    gap: 14px;
  }
  @media (max-width: 1024px) { .bottom-grid { grid-template-columns: minmax(0, 1fr); } }

  table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
  th, td { padding: 7px 8px; border-bottom: 1px solid var(--border); text-align: center; }
  th { background: #f1f5f9; font-weight: 700; color: #334155; position: sticky; top: 0; z-index: 1; }
  tr.highlight-row { background: #eff6ff; font-weight: 600; }
  .insight-box {
    background: #fffbeb; border-left: 4px solid #d97706;
    padding: 10px 12px; border-radius: 6px; font-size: 0.85rem; margin-top: 10px;
  }
  .log-box {
    height: 210px; overflow-y: auto; font-size: 0.8rem;
    border: 1px solid var(--border); border-radius: 8px; margin-top: 10px;
  }
  .log-box table th, .log-box table td { padding: 5px 7px; font-size: 0.77rem; }
  .footnote { font-size: 0.78rem; color: var(--muted); margin-top: 8px; }
</style>
</head>
<body>
<header>
  <div>
    <h1>PHÂN BỔ TÁC VỤ &amp; CÂN BẰNG TẢI TRÊN 3 VM — DEMO TRỰC QUAN HEFT vs. CPOP</h1>
    <p>Bài tập lớn Thuật toán Nâng cao | Đại học Thủy Lợi | Giả lập cụm VM_01 (.11) · VM_02 (.12) · VM_03 (.13)</p>
  </div>
  <div class="badge-header">Python Standard Library · 1 đơn vị mô hình = 10 ms (HTTP LAB)</div>
</header>

<div class="container">
  <!-- THANH DIEU KHIEN -->
  <div class="panel">
    <div class="controls-row">
      <label>Bộ dữ liệu DAG:</label>
      <select id="modeSelect">
        <option value="paper2002">Chuẩn IEEE 2002 — Fig. 3 (10 tác vụ: HEFT=80, CPOP=86)</option>
        <option value="random">DAG ngẫu nhiên (tùy chỉnh quy mô 6–100 &amp; CCR)</option>
      </select>

      <span id="randomParams" style="display:none; align-items:center; gap:8px; flex-wrap:wrap;">
        <label>Tác vụ:</label>
        <input id="tasksInput" type="number" min="6" max="100" value="30" style="width:68px;" />
        <label>CCR:</label>
        <select id="ccrInput">
          <option value="0.1">0.1 (Tính toán)</option>
          <option value="1.0" selected>1.0 (Cân bằng)</option>
          <option value="5.0">5.0 (Truyền thông cao)</option>
        </select>
        <label>Density:</label>
        <input id="densityInput" type="number" step="0.05" min="0.05" max="0.6" value="0.15" style="width:68px;" />
        <label>Seed:</label>
        <input id="seedInput" type="number" value="2026" style="width:80px;" />
      </span>

      <button class="primary" onclick="loadData()">Tải &amp; Lập lịch</button>
      <span id="dagLabel" style="font-size:0.83rem; color:#475569; font-weight:600; margin-left:auto;"></span>
    </div>

    <div class="playback-bar">
      <button id="playBtn" class="success" onclick="togglePlay()">▶ Chạy đua 3 VM</button>
      <button onclick="resetPlayback()">↺ Đặt lại (t = 0)</button>
      <button onclick="stepTime(-1)">−1</button>
      <button onclick="stepTime(1)">+1</button>
      <div class="clock-box" id="clockBox">t = 0.0 / 86.0</div>
      <input id="timeSlider" type="range" min="0" max="86" step="0.5" value="0" oninput="onSlide(this.value)" />
      <label>Tốc độ:</label>
      <select id="speedSelect">
        <option value="6">0.5x (Chậm để giảng)</option>
        <option value="14" selected>1x (Chuẩn)</option>
        <option value="35">2.5x (Nhanh)</option>
        <option value="120">8x (Cho 60–100 tác vụ)</option>
      </select>
    </div>
  </div>

  <!-- KHU VUC CHAY DUA SONG SONG HEFT vs CPOP -->
  <div class="race-grid">
    <!-- COT TRAI: HEFT -->
    <div class="panel">
      <div class="algo-header">
        <div class="algo-title">
          <span class="pill pill-heft">HEFT (AVCT)</span>
          <span class="algo-subtitle" title="Ưu tiên rank_u giảm dần + chọn máy có EFT nhỏ nhất">Ưu tiên rank_u + chọn EFT nhỏ nhất</span>
        </div>
        <span id="heftStatus" class="pill pill-heft">Sẵn sàng</span>
      </div>
      <div class="vm-grid" id="heftVMs"></div>
      <div class="gantt-wrap" id="heftGantt"></div>
      <div class="metrics-strip" id="heftMetrics"></div>
      <div class="log-box" id="heftLog"></div>
    </div>

    <!-- COT PHAI: CPOP -->
    <div class="panel">
      <div class="algo-header">
        <div class="algo-title">
          <span class="pill pill-cpop">CPOP</span>
          <span id="cpopSubTitle" class="algo-subtitle">Gom đường găng lên 1 máy</span>
        </div>
        <span id="cpopStatus" class="pill pill-cpop">Sẵn sàng</span>
      </div>
      <div class="vm-grid" id="cpopVMs"></div>
      <div class="gantt-wrap" id="cpopGantt"></div>
      <div class="metrics-strip" id="cpopMetrics"></div>
      <div class="log-box" id="cpopLog"></div>
    </div>
  </div>

  <!-- SO DO DAG & BANG DOI CHIEU 5 THUAT TOAN -->
  <div class="bottom-grid">
    <div class="panel">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px;">
        <strong>Sơ đồ đồ thị tác vụ DAG (đồng bộ theo HEFT &amp; đường găng CPOP)</strong>
        <span id="dagHint" style="font-size:0.75rem; color:#64748b; white-space:nowrap;">Viền đứt cam = Đường găng CPOP</span>
      </div>
      <div id="dagSvgWrap" style="width:100%; overflow-x:auto;"></div>
      <div id="insightBox" class="insight-box"></div>
    </div>

    <div class="panel">
      <strong style="display:block; margin-bottom:8px;">Bảng đối chiếu đầy đủ 5 thuật toán trên cùng đầu vào</strong>
      <div style="overflow-x:auto;">
        <table id="compareTable">
          <thead>
            <tr>
              <th>Thuật toán</th>
              <th>Makespan</th>
              <th>SLR</th>
              <th>Speedup</th>
              <th>CV_load</th>
              <th>Utilization (P1 / P2 / P3)</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="footnote">
        * Ghi chú học thuật: <b>AVBS*</b> duyệt khe theo thời lượng tĩnh (cho kết quả trùng HEFT).
        <b>Utilization</b> và <b>CV_load</b> tính theo lịch mô hình, không phải %CPU hệ điều hành.
        Di chuột vào từng khối trên biểu đồ Gantt hoặc nút trên DAG để xem chi tiết từng tác vụ.
      </div>
    </div>
  </div>
</div>

<script>
const VM_META = [
  { proc: "P1", vm: "VM_01", ip: "192.168.241.11", spec: "3 vCPU · 3GB", color: "#1d4ed8", cls: "active-p1" },
  { proc: "P2", vm: "VM_02", ip: "192.168.241.12", spec: "2 vCPU · 3GB", color: "#c2410c", cls: "active-p2" },
  { proc: "P3", vm: "VM_03", ip: "192.168.241.13", spec: "1 vCPU · 3GB", color: "#0f766e", cls: "active-p3" },
];

let DATA = null;
let currentTime = 0;
let maxTime = 86;
let playing = false;
let rafId = null;
let lastTs = null;

document.getElementById("modeSelect").addEventListener("change", (e) => {
  document.getElementById("randomParams").style.display = e.target.value === "random" ? "flex" : "none";
});

function formatCriticalPathShort(cp, proc) {
  if (cp.length <= 5) {
    return `Đường găng (${cp.join("→")}) trên ${proc}`;
  }
  return `Đường găng (${cp.length} nút: ${cp[0]}→${cp[1]}→…→${cp[cp.length - 1]}) trên ${proc}`;
}

async function loadData() {
  pausePlay();
  const mode = document.getElementById("modeSelect").value;
  let url = `/api/schedule?mode=${mode}`;
  if (mode === "random") {
    let tasks = parseInt(document.getElementById("tasksInput").value || "30", 10);
    tasks = Math.max(6, Math.min(100, tasks));
    document.getElementById("tasksInput").value = tasks;
    const ccr = document.getElementById("ccrInput").value;
    const density = document.getElementById("densityInput").value;
    const seed = document.getElementById("seedInput").value;
    url += `&tasks=${tasks}&ccr=${ccr}&density=${density}&seed=${seed}`;
  }
  const res = await fetch(url);
  DATA = await res.json();

  const hM = DATA.results.heft.metrics.makespan;
  const cM = DATA.results.cpop.metrics.makespan;
  maxTime = Math.max(hM, cM, 1);

  document.getElementById("dagLabel").textContent = DATA.dag.name;
  const cp = DATA.results.cpop.critical_path;
  const cpProc = DATA.results.cpop.cp_processor;
  const subEl = document.getElementById("cpopSubTitle");
  subEl.textContent = formatCriticalPathShort(cp, cpProc);
  subEl.title = `Toàn bộ đường găng (${cp.length} tác vụ trên ${cpProc}): ${cp.join(" → ")}`;

  document.getElementById("timeSlider").max = maxTime;
  document.getElementById("timeSlider").step = maxTime > 300 ? "1" : "0.2";
  currentTime = 0;
  document.getElementById("timeSlider").value = 0;

  // Tu dong chon toc do phu hop neu do thi lon
  if (DATA.dag.tasks.length >= 45 && parseFloat(document.getElementById("speedSelect").value) < 35) {
    document.getElementById("speedSelect").value = "120";
  }

  renderStaticPanels();
  buildStaticDAG();
  renderFrame(0);
}

function renderStaticPanels() {
  renderMetricsStrip("heftMetrics", DATA.results.heft.metrics);
  renderMetricsStrip("cpopMetrics", DATA.results.cpop.metrics);
  renderLogTable("heftLog", DATA.results.heft.placements);
  renderLogTable("cpopLog", DATA.results.cpop.placements);
  renderCompareTable();
  renderInsight();
}

function renderMetricsStrip(elId, m) {
  document.getElementById(elId).innerHTML = `
    <div class="metric-box"><div class="lbl">Makespan</div><div class="val">${m.makespan}</div></div>
    <div class="metric-box"><div class="lbl">SLR (LB=${m.lb})</div><div class="val">${m.slr.toFixed(4)}</div></div>
    <div class="metric-box"><div class="lbl">Speedup</div><div class="val">${m.speedup.toFixed(4)}</div></div>
    <div class="metric-box"><div class="lbl">CV_load</div><div class="val">${m.cv_load.toFixed(4)}</div></div>
  `;
}

function renderLogTable(elId, placements) {
  const rows = placements.map(p => `
    <tr>
      <td>${p.step}</td>
      <td><b>${p.task}</b>${p.is_cp ? ' ★' : ''}</td>
      <td>${p.proc}</td>
      <td style="white-space:nowrap;">${p.start} – ${p.finish}</td>
      <td style="text-align:left;">${p.reason}</td>
    </tr>
  `).join("");
  document.getElementById(elId).innerHTML = `
    <table>
      <thead><tr><th>#</th><th>Tác vụ</th><th>Máy</th><th>AST–AFT</th><th>Quyết định chọn máy</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderCompareTable() {
  const labels = { heft: "HEFT (AVCT)", cpop: "CPOP", mxct: "MXCT", mnct: "MNCT", avbs: "AVBS*" };
  const tbody = document.querySelector("#compareTable tbody");
  const minM = Math.min(...Object.values(DATA.results).map(r => r.metrics.makespan));
  tbody.innerHTML = ["heft", "cpop", "mxct", "mnct", "avbs"].map(k => {
    const m = DATA.results[k].metrics;
    const isBest = Math.abs(m.makespan - minM) < 1e-6;
    return `
      <tr class="${isBest ? 'highlight-row' : ''}">
        <td>${labels[k]}</td>
        <td>${m.makespan}</td>
        <td>${m.slr.toFixed(4)}</td>
        <td>${m.speedup.toFixed(4)}</td>
        <td>${m.cv_load.toFixed(4)}</td>
        <td>${m.utilization.map(u => u.toFixed(1) + '%').join(' / ')}</td>
      </tr>
    `;
  }).join("");
}

function renderInsight() {
  const h = DATA.results.heft.metrics;
  const c = DATA.results.cpop.metrics;
  const diff = c.makespan - h.makespan;
  const pct = c.makespan > 0 ? ((Math.abs(diff) / Math.max(c.makespan, h.makespan)) * 100).toFixed(2) : "0";
  let msg = "";
  if (diff > 0) {
    msg = `<b>Điểm nhấn trình bày:</b> Trên đồ thị này, <b>HEFT (${h.makespan})</b> hoàn thành sớm hơn <b>CPOP (${c.makespan})</b> là <b>${diff.toFixed(1)} đơn vị (${pct}%)</b> `
        + `(tương đương <b>${(diff*10).toFixed(0)} ms</b> ở mức 10 ms/đơn vị). `
        + `Độ lệch tải CV_load của HEFT là <b>${h.cv_load.toFixed(4)}</b> so với <b>${c.cv_load.toFixed(4)}</b> của CPOP `
        + `(CPOP gom đường găng gồm ${DATA.results.cpop.critical_path.length} nút lên <b>${DATA.results.cpop.cp_processor}</b>).`;
  } else if (diff < 0) {
    msg = `<b>Điểm nhấn trình bày:</b> Trường hợp minh họa <b>CPOP (${c.makespan}) thắng HEFT (${h.makespan})</b> sớm hơn <b>${Math.abs(diff).toFixed(1)} đơn vị (${pct}%)</b>! `
        + `Khi chi phí truyền thông lớn, việc CPOP gom chuỗi đường găng lên <b>${DATA.results.cpop.cp_processor}</b> giúp giảm đáng kể trễ truyền thông.`;
  } else {
    msg = `<b>Điểm nhấn trình bày:</b> HEFT và CPOP đạt cùng Makespan = <b>${h.makespan}</b> trên cấu hình DAG này.`;
  }
  document.getElementById("insightBox").innerHTML = msg;
}

function renderVMCards(containerId, resObj, t) {
  const html = VM_META.map((vm, j) => {
    const running = resObj.placements.find(p => p.proc_idx === j && t >= p.start && t < p.finish);
    const doneCount = resObj.placements.filter(p => p.proc_idx === j && t >= p.finish).length;
    const totalCount = resObj.placements.filter(p => p.proc_idx === j).length;
    const pct = running ? Math.min(100, ((t - running.start) / running.duration) * 100) : 0;
    const activeCls = running ? vm.cls : "";
    const stateText = running
      ? `RUN ${running.task} (${running.start}→${running.finish})`
      : (t >= resObj.metrics.makespan ? `DONE (${doneCount}/${totalCount})` : `IDLE (${doneCount}/${totalCount})`);
    return `
      <div class="vm-card ${activeCls}">
        <div class="vm-top">
          <span style="color:${vm.color}">${vm.vm} (${vm.proc})</span>
          <span class="vm-spec">${vm.spec}</span>
        </div>
        <div class="vm-ip">${vm.ip}:8000</div>
        <div class="vm-state ${running ? 'running' : ''}" title="${stateText}">${stateText}</div>
        <div class="vm-bar-wrap">
          <div class="vm-bar" style="width:${pct}%; background:${vm.color};"></div>
        </div>
      </div>
    `;
  }).join("");
  document.getElementById(containerId).innerHTML = html;
}

function renderGantt(containerId, resObj, t) {
  const W = 620, H = 160, padL = 42, padR = 16, padT = 16, padB = 26;
  const plotW = W - padL - padR;
  const rowH = 34, gap = 8;
  const xOf = (val) => padL + (val / maxTime) * plotW;
  const manyTasks = resObj.placements.length > 25;

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block; background:#fcfdff; border:1px solid #e2e8f0; border-radius:8px;">`;
  const ticks = 6;
  for (let i = 0; i <= ticks; i++) {
    const v = (maxTime * i) / ticks;
    const x = xOf(v);
    svg += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${H-padB}" stroke="#e2e8f0" stroke-dasharray="3,3"/>`;
    svg += `<text x="${x}" y="${H-8}" font-size="10" fill="#64748b" text-anchor="middle">${v.toFixed(0)}</text>`;
  }

  VM_META.forEach((vm, j) => {
    const y = padT + j * (rowH + gap);
    svg += `<rect x="${padL}" y="${y}" width="${plotW}" height="${rowH}" fill="#f8fafc" rx="4"/>`;
    svg += `<text x="${padL - 8}" y="${y + rowH/2 + 4}" font-size="12" font-weight="700" fill="${vm.color}" text-anchor="end">${vm.proc}</text>`;
  });

  resObj.placements.forEach(p => {
    const y = padT + p.proc_idx * (rowH + gap) + 3;
    const x = xOf(p.start);
    const fullW = Math.max(1.5, xOf(p.finish) - x);
    const col = VM_META[p.proc_idx].color;
    const tip = `${p.task} | ${p.proc} | ${p.start} → ${p.finish} (thời lượng: ${p.duration})${p.is_cp ? ' [Đường găng]' : ''}`;

    svg += `<g><title>${tip}</title>`;
    svg += `<rect x="${x}" y="${y}" width="${fullW}" height="${rowH-6}" rx="${manyTasks ? 2 : 5}" fill="${col}" fill-opacity="0.18" stroke="${col}" stroke-opacity="0.4" stroke-width="0.8"/>`;

    if (t > p.start) {
      const curEnd = Math.min(t, p.finish);
      const curW = Math.max(1, xOf(curEnd) - x);
      svg += `<rect x="${x}" y="${y}" width="${curW}" height="${rowH-6}" rx="${manyTasks ? 2 : 5}" fill="${col}"/>`;
    }
    if (p.is_cp) {
      svg += `<rect x="${x}" y="${y}" width="${fullW}" height="${rowH-6}" rx="${manyTasks ? 2 : 5}" fill="none" stroke="#facc15" stroke-width="${manyTasks ? 1.1 : 2}"/>`;
    }
    if (fullW >= 24) {
      const fSize = fullW < 34 ? 9 : 11;
      const txtCol = t >= (p.start + p.finish) / 2 ? "#ffffff" : "#0f172a";
      svg += `<text x="${x + fullW/2}" y="${y + (rowH-6)/2 + 3.5}" font-size="${fSize}" font-weight="700" fill="${txtCol}" text-anchor="middle">${p.task}</text>`;
    }
    svg += `</g>`;
  });

  const px = xOf(Math.min(t, maxTime));
  svg += `<line x1="${px}" y1="${padT-6}" x2="${px}" y2="${H-padB+4}" stroke="#dc2626" stroke-width="2"/>`;
  svg += `<circle cx="${px}" cy="${padT-6}" r="3.5" fill="#dc2626"/>`;
  svg += `</svg>`;
  document.getElementById(containerId).innerHTML = svg;
}

// Xay dung so do DAG 1 lan duy nhat khi tai du lieu, sau do chi cap nhat mau nut (60fps muot cho 100 tac vu)
function buildStaticDAG() {
  const dag = DATA.dag;
  const levels = DATA.levels;
  const nTasks = dag.tasks.length;
  const cpSet = new Set(DATA.results.cpop.critical_path);
  const cpEdges = new Set();
  for (let i = 0; i < DATA.results.cpop.critical_path.length - 1; i++) {
    cpEdges.add(DATA.results.cpop.critical_path[i] + "->" + DATA.results.cpop.critical_path[i + 1]);
  }

  const byLvl = {};
  let maxLvl = 0;
  let maxPerLvl = 1;
  dag.tasks.forEach(tk => {
    const lv = levels[tk] || 0;
    maxLvl = Math.max(maxLvl, lv);
    if (!byLvl[lv]) byLvl[lv] = [];
    byLvl[lv].push(tk);
    maxPerLvl = Math.max(maxPerLvl, byLvl[lv].length);
  });

  const W = 620;
  const H = nTasks > 35 ? Math.min(360, Math.max(260, maxPerLvl * 26)) : 260;
  const r = nTasks <= 15 ? 15 : (nTasks <= 35 ? 10.5 : 7);
  const padX = 28, padY = 22;
  const pos = {};

  for (let lv = 0; lv <= maxLvl; lv++) {
    const arr = byLvl[lv] || [];
    const x = maxLvl === 0 ? W / 2 : padX + (lv / maxLvl) * (W - 2 * padX);
    arr.forEach((tk, idx) => {
      const y = arr.length === 1 ? H / 2 : padY + (idx / (arr.length - 1)) * (H - 2 * padY);
      pos[tk] = { x, y };
    });
  }

  document.getElementById("dagHint").textContent = nTasks > 35
    ? `Đồ thị lớn (${nTasks} nút) · Di chuột vào nút xem chi tiết`
    : "Viền cam = Đường găng CPOP";

  const heftMap = {};
  DATA.results.heft.placements.forEach(p => heftMap[p.task] = p);

  let svg = `<svg id="dagSvg" viewBox="0 0 ${W} ${H}" width="100%" style="display:block; background:#fcfdff; border:1px solid #e2e8f0; border-radius:8px;">`;
  if (nTasks <= 35) {
    svg += `<defs><marker id="arr" viewBox="0 0 10 10" refX="${r + 4}" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#94a3b8"/></marker></defs>`;
  }

  // Ve cac canh thuong truoc, canh duong gang ve de len sau cho noi bat
  dag.edges.forEach(e => {
    const isCp = cpEdges.has(e.src + "->" + e.dst);
    if (isCp) return;
    const a = pos[e.src], b = pos[e.dst];
    const op = nTasks > 35 ? 0.25 : 0.75;
    svg += `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="#cbd5e1" stroke-opacity="${op}" stroke-width="${nTasks > 35 ? 0.8 : 1.2}" ${nTasks <= 35 ? 'marker-end="url(#arr)"' : ''}/>`;
    if (nTasks <= 14) {
      svg += `<text x="${(a.x+b.x)/2}" y="${(a.y+b.y)/2 - 3}" font-size="9" fill="#64748b" text-anchor="middle">${e.comm}</text>`;
    }
  });
  dag.edges.forEach(e => {
    const isCp = cpEdges.has(e.src + "->" + e.dst);
    if (!isCp) return;
    const a = pos[e.src], b = pos[e.dst];
    svg += `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="#f59e0b" stroke-width="${nTasks > 35 ? 1.8 : 2.4}" stroke-dasharray="5,3" ${nTasks <= 35 ? 'marker-end="url(#arr)"' : ''}/>`;
  });

  dag.tasks.forEach(tk => {
    const { x, y } = pos[tk];
    const pl = heftMap[tk];
    const isCp = cpSet.has(tk);
    const tip = `${tk} | HEFT: ${pl.proc} (${pl.start}→${pl.finish})${isCp ? ' | Thuộc đường găng CPOP' : ''}`;
    svg += `<g><title>${tip}</title>`;
    svg += `<circle id="dag-node-${tk}" cx="${x}" cy="${y}" r="${r}" fill="#f1f5f9" stroke="${isCp ? '#d97706' : '#64748b'}" stroke-width="${isCp ? 2.4 : 1.2}"/>`;
    if (nTasks <= 35) {
      svg += `<text id="dag-txt-${tk}" x="${x}" y="${y + (nTasks <= 15 ? 3.8 : 3)}" font-size="${nTasks <= 15 ? 10 : 8}" font-weight="700" fill="#0f172a" text-anchor="middle">${tk}</text>`;
    }
    svg += `</g>`;
  });

  svg += `</svg>`;
  document.getElementById("dagSvgWrap").innerHTML = svg;
}

function updateDAGColors(t) {
  const cpSet = new Set(DATA.results.cpop.critical_path);
  DATA.results.heft.placements.forEach(pl => {
    const circle = document.getElementById(`dag-node-${pl.task}`);
    if (!circle) return;
    const txt = document.getElementById(`dag-txt-${pl.task}`);
    const isCp = cpSet.has(pl.task);
    if (t >= pl.finish) {
      const col = VM_META[pl.proc_idx].color;
      circle.setAttribute("fill", col);
      circle.setAttribute("stroke", isCp ? "#f59e0b" : col);
      if (txt) txt.setAttribute("fill", "#ffffff");
    } else if (t >= pl.start) {
      circle.setAttribute("fill", "#facc15");
      circle.setAttribute("stroke", "#b45309");
      if (txt) txt.setAttribute("fill", "#0f172a");
    } else {
      circle.setAttribute("fill", "#f1f5f9");
      circle.setAttribute("stroke", isCp ? "#d97706" : "#64748b");
      if (txt) txt.setAttribute("fill", "#0f172a");
    }
  });
}

function renderFrame(t) {
  if (!DATA) return;
  document.getElementById("clockBox").textContent = `t = ${t.toFixed(1)} / ${maxTime.toFixed(1)}`;
  document.getElementById("timeSlider").value = t;

  const hRes = DATA.results.heft;
  const cRes = DATA.results.cpop;

  renderVMCards("heftVMs", hRes, t);
  renderVMCards("cpopVMs", cRes, t);
  renderGantt("heftGantt", hRes, t);
  renderGantt("cpopGantt", cRes, t);
  updateDAGColors(t);

  const hSt = document.getElementById("heftStatus");
  if (t >= hRes.metrics.makespan) {
    hSt.textContent = `✓ Xong (t=${hRes.metrics.makespan})`;
    hSt.className = "pill pill-done";
  } else if (t > 0) {
    hSt.textContent = "Đang chạy...";
    hSt.className = "pill pill-run";
  } else {
    hSt.textContent = "Sẵn sàng";
    hSt.className = "pill pill-heft";
  }

  const cSt = document.getElementById("cpopStatus");
  if (t >= cRes.metrics.makespan) {
    cSt.textContent = `✓ Xong (t=${cRes.metrics.makespan})`;
    cSt.className = "pill pill-done";
  } else if (t > 0) {
    cSt.textContent = "Đang chạy...";
    cSt.className = "pill pill-run";
  } else {
    cSt.textContent = "Sẵn sàng";
    cSt.className = "pill pill-cpop";
  }
}

function tick(ts) {
  if (!playing) return;
  if (lastTs === null) lastTs = ts;
  const dtSec = (ts - lastTs) / 1000.0;
  lastTs = ts;
  const speed = parseFloat(document.getElementById("speedSelect").value);
  currentTime = Math.min(maxTime, currentTime + dtSec * speed);
  renderFrame(currentTime);
  if (currentTime >= maxTime) {
    pausePlay();
  } else {
    rafId = requestAnimationFrame(tick);
  }
}

function togglePlay() {
  if (playing) {
    pausePlay();
  } else {
    if (currentTime >= maxTime) currentTime = 0;
    playing = true;
    lastTs = null;
    document.getElementById("playBtn").textContent = "⏸ Tạm dừng";
    rafId = requestAnimationFrame(tick);
  }
}

function pausePlay() {
  playing = false;
  if (rafId) cancelAnimationFrame(rafId);
  document.getElementById("playBtn").textContent = "▶ Chạy đua 3 VM";
}

function resetPlayback() {
  pausePlay();
  currentTime = 0;
  renderFrame(0);
}

function stepTime(delta) {
  pausePlay();
  currentTime = Math.max(0, Math.min(maxTime, Math.round(currentTime + delta)));
  renderFrame(currentTime);
}

function onSlide(val) {
  pausePlay();
  currentTime = parseFloat(val);
  renderFrame(currentTime);
}

loadData();
</script>
</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/schedule":
            qs = parse_qs(parsed.query)
            mode = qs.get("mode", ["paper2002"])[0]
            if mode == "random":
                tasks = int(qs.get("tasks", ["30"])[0])
                ccr = float(qs.get("ccr", ["1.0"])[0])
                density = float(qs.get("density", ["0.15"])[0])
                seed = int(qs.get("seed", ["2026"])[0])
                dag = generate_random_dag(tasks, ccr, density, seed)
            else:
                dag = get_paper2002_dag()
            payload = json.dumps(evaluate_all(dag), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser(description="Web Demo 3-VM Simulator (HEFT vs CPOP)")
    parser.add_argument("--host", default="127.0.0.1", help="Dia chi bind (mac dinh 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5050, help="Cong HTTP (mac dinh 5050)")
    parser.add_argument("--no-browser", action="store_true", help="Khong tu dong mo trinh duyet")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"[OK] Web Demo dang chay tai: {url}")
    print("Nhan Ctrl+C de dung server.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDa dung Web Demo.")
        server.server_close()


if __name__ == "__main__":
    main()