import itertools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace"))

from round6_tail import action, apply_known, deterministic_d5_suffix, evaluate_sequence, tail_plan


def state(nodes, edges, effects):
    return nodes, set(edges), dict(effects), {}


class Round6TailTests(unittest.TestCase):
    def test_unknown_d5_response_abstains(self):
        nodes = {
            1: {"w": 0.0, "persona": "和平", "comm_left": 3},
            2: {"w": 0.0, "persona": "和平", "comm_left": 3},
        }
        result = deterministic_d5_suffix(state(nodes, {(1, 2)}, {}), 2.0, 1)
        self.assertEqual(result["status"], "unknown_response")
        planned = tail_plan(state(nodes, {(1, 2)}, {}), 2.0, 1)
        self.assertEqual(planned["status"], "abstain")

    def test_known_communication_uses_public_effect_and_decay(self):
        nodes = {1: {"w": 0.0, "persona": "中立", "comm_left": 2}}
        current = state(nodes, set(), {1: 8.0})
        after = apply_known(current, action("comm", 1))
        self.assertEqual(after[0][1]["w"], 4.0)
        self.assertEqual(after[0][1]["comm_left"], 1)
        again = apply_known(after, action("comm", 1))
        self.assertEqual(again[0][1]["w"], 6.0)

    def test_tail_value_matches_exhaustive_legal_sequences(self):
        nodes = {
            1: {"w": -30.0, "persona": "暴力", "comm_left": 1},
            2: {"w": 0.0, "persona": "中立", "comm_left": 2},
            3: {"w": 20.0, "persona": "和平", "comm_left": 2},
            4: {"w": -5.0, "persona": "中立", "comm_left": 1},
        }
        current = state(nodes, {(1, 2), (2, 3), (3, 4)}, {1: 1.0, 2: 8.0, 3: 12.0, 4: 4.0})
        result = tail_plan(current, 6.0, 3, improvement_margin=-1e9, structure_limit=512)
        primitive = [action("comm", i) for i in nodes]
        primitive += [action("cut", *edge) for edge in current[1]]
        primitive += [action("shield", i) for i in nodes]
        values = []
        for length in range(4):
            for sequence in itertools.product(primitive, repeat=length):
                if sum(item["cost"] for item in sequence) > 6:
                    continue
                try:
                    value, _ = evaluate_sequence(current, sequence)
                except ValueError:
                    continue
                values.append(value)
        self.assertAlmostEqual(result["candidate_score"], max(values), places=9)

    def test_structure_limit_reports_truncation(self):
        nodes = {i: {"w": float(i), "persona": "中立", "comm_left": 0} for i in range(6)}
        current = state(nodes, {(i, i + 1) for i in range(5)}, {})
        result = tail_plan(current, 6.0, 3, structure_limit=1)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["evaluated_structures"], 1)


if __name__ == "__main__":
    unittest.main()
