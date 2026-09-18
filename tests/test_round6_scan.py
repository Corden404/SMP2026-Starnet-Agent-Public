import copy
import sys
import unittest
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace"))

from experiment_round3 import fast_model
from experiment_round5 import FiniteEnv, PublicEnv
from mechanism_round5 import seed
from round6_common import people
from round6_scan import S42, initialize_fast_scan


class Round6ScanTests(unittest.TestCase):
    def test_fixed_stop_uses_only_public_scan_count(self):
        graph = nx.path_graph(range(1, 51))
        raw = FiniteEnv(seed(graph, {i: 0.0 for i in graph}, r=1.0))
        model = fast_model(PublicEnv(raw), copy.deepcopy(people()), S42)
        initialize_fast_scan(model)
        while len(model.local_nodes) < 42:
            model.step()
        self.assertFalse(model._should_scan(raw.get_remaining_budget()))
        self.assertIsNone(model._next_scan_target())
        self.assertEqual(len(model.local_nodes), 42)
        self.assertEqual(model.scan_diagnostic_records[-1]["unknown_nodes"], 8)


if __name__ == "__main__":
    unittest.main()
