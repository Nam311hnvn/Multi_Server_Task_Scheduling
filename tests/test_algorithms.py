"""Run: python -m unittest discover -s tests -v"""
import json
from pathlib import Path
import unittest
from task_scheduler.model import TaskGraph
from task_scheduler.algorithms import schedule, upward_ranks, downward_ranks, validate_schedule
from task_scheduler.metrics import evaluate
from task_scheduler.generate import random_graph

HERE = Path(__file__).resolve().parents[1]


class SchedulingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paper = TaskGraph.from_dict(json.loads((HERE / 'examples' / 'topcuoglu2002_fig3.json').read_text(encoding='utf8')))

    def test_paper_upward_rank(self):
        ranks = upward_ranks(self.paper)
        self.assertAlmostEqual(ranks['n1'], 108, places=2)
        self.assertAlmostEqual(ranks['n2'], 77, places=2)
        self.assertAlmostEqual(ranks['n10'], 44/3, places=2)

    def test_paper_heft_makespan_80(self):
        self.assertAlmostEqual(schedule(self.paper, 'heft').makespan, 80)

    def test_paper_cpop_makespan_86(self):
        result = schedule(self.paper, 'cpop')
        self.assertEqual(result.critical_path, ['n1', 'n2', 'n9', 'n10'])
        self.assertEqual(result.cp_machine, 'P2')
        self.assertAlmostEqual(result.makespan, 86)

    def test_2023_variants_have_valid_schedules(self):
        for graph in [self.paper, random_graph(20, seed=6), random_graph(35, density=0.1, seed=7)]:
            for method in ['heft', 'cpop', 'mxct', 'mnct', 'avbs']:
                sched = schedule(graph, method)
                validate_schedule(graph, sched)
                self.assertGreaterEqual(evaluate(graph, sched)['slr'], 1-1e-8)

    def test_avbs_same_under_standard_heft_assumptions(self):
        for seed in range(15):
            graph = random_graph(15, seed=seed)
            self.assertAlmostEqual(schedule(graph, 'heft').makespan, schedule(graph, 'avbs').makespan)

    def test_reject_cycle(self):
        with self.assertRaisesRegex(ValueError, 'cycle'):
            TaskGraph.from_dict({'machines': ['p'], 'tasks': [{'id': 'A', 'costs': [1]}, {'id': 'B', 'costs': [1]}],
                                 'edges': [{'from': 'A', 'to': 'B', 'comm': 1}, {'from': 'B', 'to': 'A', 'comm': 1}]})

    def test_serial_chain(self):
        graph = TaskGraph.from_dict({'machines': ['fast', 'slow'], 'tasks': [{'id': 'A', 'costs': [2, 10]}, {'id': 'B', 'costs': [3, 10]}],
                                     'edges': [{'from': 'A', 'to': 'B', 'comm': 10}]})
        self.assertEqual(schedule(graph, 'heft').makespan, 5)


if __name__ == '__main__':
    unittest.main()
