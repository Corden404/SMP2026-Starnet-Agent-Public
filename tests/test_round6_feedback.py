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
from round6_feedback import L1, L3, U3, VARIANTS, initialize_fast_feedback


class Round6FeedbackTests(unittest.TestCase):
    def make_model(self, cls):
        raw = FiniteEnv(seed(nx.path_graph(range(1, 5)), {i: 0.0 for i in range(1, 5)}, r=1.0))
        model = fast_model(PublicEnv(raw), copy.deepcopy(people()), cls)
        initialize_fast_feedback(model)
        for node in range(1, 5):
            model._execute_scan(node)
        return model

    def test_four_cell_options_are_single_variable_changes(self):
        self.assertEqual({VARIANTS[name]["rollout_width"] for name in VARIANTS}, {2})
        self.assertEqual((VARIANTS["L1"]["rollout_learning"], VARIANTS["L1"]["rollout_samples"]), (True, 1))
        self.assertEqual((VARIANTS["U3"]["rollout_learning"], VARIANTS["U3"]["rollout_samples"]), (False, 3))
        self.assertEqual((VARIANTS["L3"]["rollout_learning"], VARIANTS["L3"]["rollout_samples"]), (True, 3))

    def test_production_candidate_entry_records_active_variant(self):
        for cls, name in ((L1, "L1"), (U3, "U3"), (L3, "L3")):
            model = self.make_model(cls)
            candidates = model._learned_candidates(model.env.get_remaining_budget())
            self.assertTrue(candidates)
            record = model.feedback_records[-1]
            self.assertEqual(record["variant"], name)
            self.assertEqual(record["learning"], VARIANTS[name]["rollout_learning"])
            self.assertEqual(record["samples"], VARIANTS[name]["rollout_samples"])

    def test_candidate_search_does_not_mutate_real_observations(self):
        model = self.make_model(L3)
        before_nodes = copy.deepcopy(model.local_nodes)
        before_effects = copy.deepcopy(model.comm_effects)
        before_observations = copy.deepcopy(model.persona_observations)
        model._learned_candidates(model.env.get_remaining_budget())
        self.assertEqual(model.local_nodes, before_nodes)
        self.assertEqual(model.comm_effects, before_effects)
        self.assertEqual(model.persona_observations, before_observations)


if __name__ == "__main__":
    unittest.main()
