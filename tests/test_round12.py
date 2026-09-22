"""No network: decision-sensitive scans must pay for both information and slots."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workspace'))
import round12_experiment as study
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv
from round4_seeds import generate


def shield(i):
    return dict(action='shield', target_node_1=i, target_node_2=None,
                prompt_id=None, cost=5., gain=10., roi=2.)


class Round12Tests(unittest.TestCase):
    def model(self, variant='V2', index=0, mode='fast'):
        module = study.load(variant)
        raw = FiniteEnv(generate(study.specs('dev')[index]))
        logged = LoggedEnv(raw)
        with contextlib.redirect_stdout(io.StringIO()):
            model = study.initialize(module, logged, mode=mode)
            for _ in range(44):
                self.assertEqual(model.step(), 0)
            self.assertFalse(model._should_scan(raw.get_remaining_budget()))
        return module, raw, logged, model

    def test_exact_standalone_package_without_oracles_or_f1(self):
        original = study.base.source_files('S44')
        self.assertEqual(study.source_files('S44'), original)
        candidate = study.source_files('V2')
        self.assertEqual(len(candidate), 9)
        for name in candidate:
            if name == 'starnet_model.py':
                self.assertTrue(candidate[name].startswith(original[name]))
                for forbidden in (b'round12_experiment', b'round11_experiment', b'FeedbackFirst', b'true_w', b'true_edges'):
                    self.assertNotIn(forbidden, candidate[name])
            else:
                self.assertEqual(candidate[name], original[name])

    def test_first44_scans_identical_and_prior_mean_preserved(self):
        _, r0, l0, b = self.model('S44')
        _, r1, l1, c = self.model('V2')
        self.assertEqual(l0.actions, l1.actions)
        self.assertEqual(b.local_nodes, c.local_nodes)
        self.assertEqual(b.local_edges, c.local_edges)
        self.assertEqual(r0.get_remaining_budget(), 78.)
        self.assertEqual(r1.get_remaining_budget(), 78.)
        support = c._active_support()
        self.assertEqual(len(support), 2)
        self.assertAlmostEqual(sum(s['probability'] for s in support), 1.)
        self.assertAlmostEqual(sum(s['probability'] * s['w'] for s in support), c.unseen_w)
        self.assertTrue(all('r' not in d for d in c._active_prior))
        self.assertIsNone(c._next_scan_target())

    def test_prior_is_initial_scan_data_not_post_intervention_weights(self):
        _, raw, _, m = self.model()
        before = copy.deepcopy(m._active_prior)
        node = next(iter(m.local_nodes))
        m._execute_planned_action(m._action_dict('comm', node), 78., dict(m.local_nodes[node]))
        m._should_scan(raw.get_remaining_budget())
        self.assertEqual(m._active_prior, before)
        self.assertNotEqual(m.local_nodes[node]['w'], before[sorted(m.local_nodes).index(node)]['w'])

    def test_pair_is_unknown_and_ranked_by_coefficient_sensitivity(self):
        module, _, _, m = self.model()
        state = m.completed_state(); action = shield(next(iter(m.local_nodes)))
        pair = m._active_pair(state, action)
        unknown = set(state[0]) - set(m.local_nodes)
        self.assertEqual(len(pair), 2)
        self.assertTrue(set(pair) <= unknown)
        self.assertNotIn(action['target_node_1'], pair)
        _, before = module.score(state[0], state[1])
        target = action['target_node_1']
        _, after = module.score({i: d for i, d in state[0].items() if i != target},
                                {e for e in state[1] if target not in e})
        expected = sorted(unknown, key=lambda i: (-abs(after.get(i, 0.) - before[i]), i))[:2]
        self.assertEqual(pair, expected)

    def test_four_complete_scenarios_account_budget_and_slots(self):
        _, raw, logged, m = self.model()
        unknown = sorted(set(range(1, 51)) - set(m.local_nodes))
        pair = unknown[:2]; action = shield(next(iter(m.local_nodes)))
        state_before = copy.deepcopy(m.completed_state()); actions_before = copy.deepcopy(logged.actions)
        evaluations = []
        def value(state, budget, slots, eligible, deadline, first=None):
            evaluations.append((copy.deepcopy(state), budget, slots, set(eligible), first))
            return 100. if first is not None else 102.
        with patch.object(m, '_active_pair', return_value=pair), patch.object(m, '_active_value', side_effect=value):
            selected, evidence = m._active_evaluate(40., action)
        self.assertEqual(selected, pair)
        self.assertEqual(evidence['reason'], 'purchase')
        self.assertEqual(len(evaluations), 5)
        self.assertEqual(evaluations[0][1], 40.)
        slots_after = min(m.operation_limit - 1 - m.operation_count - 2,
                          m.llm_counter.safe_limit - m.llm_counter.call_count - 1)
        for state, budget, slots, eligible, first in evaluations[1:]:
            self.assertEqual(budget, 39.)
            self.assertEqual(slots, slots_after)
            self.assertEqual(eligible, set(m.local_nodes) | set(pair))
            self.assertTrue(all(state[0][i]['comm_left'] == 3 for i in pair))
            self.assertTrue(all('r' not in d for d in state[0].values()))
        self.assertAlmostEqual(sum(x['probability'] for x in evidence['scenarios']), 1.)
        self.assertEqual(m.completed_state(), state_before)
        self.assertEqual(logged.actions, actions_before)
        self.assertEqual(raw.get_remaining_budget(), 78.)

    def test_unprofitable_scan_and_incomplete_expectation_abstain(self):
        _, raw, logged, m = self.model()
        pair = sorted(set(range(1, 51)) - set(m.local_nodes))[:2]
        action = shield(next(iter(m.local_nodes)))
        before = copy.deepcopy(logged.actions)
        with patch.object(m, '_active_pair', return_value=pair), patch.object(m, '_active_value', side_effect=[100., 99., 99., 99., 99.]):
            selected, evidence = m._active_evaluate(40., action)
        self.assertIsNone(selected); self.assertEqual(evidence['reason'], 'cost_not_recovered')
        with patch.object(m, '_active_pair', return_value=pair), patch.object(m, '_active_value', side_effect=[100., 110., TimeoutError('test')]):
            selected, evidence = m._active_evaluate(40., action)
        self.assertIsNone(selected); self.assertEqual(evidence['reason'], 'incomplete_no_scan')
        self.assertEqual(logged.actions, before)
        self.assertEqual(raw.get_remaining_budget(), 78.)

    def test_zero_deadline_never_purchases(self):
        _, raw, _, m = self.model()
        pair = sorted(set(range(1, 51)) - set(m.local_nodes))[:2]
        m.active_seconds = 0.
        with patch.object(m, '_active_pair', return_value=pair):
            selected, evidence = m._active_evaluate(40., shield(next(iter(m.local_nodes))))
        self.assertIsNone(selected); self.assertEqual(evidence['reason'], 'incomplete_no_scan')
        self.assertEqual(raw.get_remaining_budget(), 78.)

    def test_two_paid_scans_use_actual_receipts_and_one_batch_call(self):
        module, raw, logged, m = self.model()
        pair = sorted(set(range(1, 51)) - set(m.local_nodes))[:2]
        prior = copy.deepcopy(m._active_prior); calls = m.llm_counter.call_count; ops = m.operation_count
        # Artificial budget is only a unit-test trigger fixture, not a scored experiment.
        raw._budget = 40.
        action = shield(next(iter(m.local_nodes)))
        with patch.object(module._ActiveBase, '_learned_candidates', return_value=[action]), \
             patch.object(m, '_active_evaluate', return_value=(pair, {'reason': 'purchase'})):
            self.assertEqual(m._planned_step(), 0)
            self.assertEqual(m._planned_step(), 0)
        self.assertEqual(raw.get_remaining_budget(), 39.)
        self.assertEqual(m.operation_count, ops + 2)
        self.assertEqual(m.llm_counter.call_count, calls + 1)
        self.assertEqual(m._active_added, 2)
        self.assertEqual(len(raw._scanned), 46)
        self.assertEqual(m._active_queue, [])
        self.assertEqual([a['args'][0] for a in logged.actions[-2:]], pair)
        self.assertEqual(m._active_prior, prior)
        self.assertIsNone(m._next_scan_target())
        self.assertFalse(m._should_scan(raw.get_remaining_budget()))
        for i in pair:
            self.assertEqual(m.local_nodes[i]['w'], raw._nodes[i]['w'])
            self.assertEqual(m.local_nodes[i]['persona'], raw._nodes[i]['persona'])
            self.assertNotIn('r', m.local_nodes[i])
        with patch.object(module._ActiveBase, '_planned_step', return_value=1), patch.object(m, '_active_evaluate') as extra:
            m._planned_step()
        extra.assert_not_called()

    def test_no_purchase_uses_cached_original_action_once(self):
        module, raw, logged, m = self.model()
        raw._budget = 40.
        action = shield(next(iter(m.local_nodes)))
        with patch.object(module._ActiveBase, '_learned_candidates', return_value=[action]) as reference, \
             patch.object(m, '_active_evaluate', return_value=(None, {'reason': 'cost_not_recovered'})):
            m._planned_step()
        self.assertEqual(reference.call_count, 1)
        self.assertEqual(logged.actions[-1]['action'], 'shield_node')
        self.assertEqual(logged.actions[-1]['args'][0], action['target_node_1'])
        self.assertEqual(raw.get_remaining_budget(), 35.)
        self.assertIsNone(m._active_reference)
        self.assertEqual(m._active_added, 0)

    def test_two_evaluation_windows_are_bounded(self):
        module, raw, _, m = self.model()
        action = shield(next(iter(m.local_nodes)))
        with patch.object(module._ActiveBase, '_learned_candidates', return_value=[action]), \
             patch.object(m, '_active_evaluate', return_value=(None, {'reason': 'cost_not_recovered'})) as evaluate, \
             patch.object(module._ActiveBase, '_planned_step', return_value=0):
            for budget in (40., 38., 25., 20., 19., 10.):
                raw._budget = budget; m._planned_step()
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(m._active_seen_windows, {40., 20.})

    def test_resource_guard_prevents_extra_environment_calls(self):
        _, raw, logged, m = self.model()
        m._active_queue = sorted(set(range(1, 51)) - set(m.local_nodes))[:2]
        m.operation_count = m.operation_limit - 1
        before = copy.deepcopy(logged.actions)
        self.assertEqual(m._planned_step(), 1)
        self.assertEqual(logged.actions, before)
        self.assertEqual(raw.get_remaining_budget(), 78.)

    def test_other_sizes_and_real_casevo_initialization(self):
        module = study.load('V2')
        for n in (4, 100):
            raw = FiniteEnv(generate(dict(study.specs('dev')[0], n=n)))
            m = study.initialize(module, raw, n=n)
            self.assertEqual(m.scan_target, n)
            with patch.object(module._ActiveBase, '_planned_step', return_value=0) as original:
                m._planned_step(); self.assertEqual(original.call_count, 1)
            self.assertEqual(m.active_records, [])
        _, _, _, fast = self.model()
        _, _, _, full = self.model(mode='integration')
        self.assertEqual(full.local_nodes, fast.local_nodes)
        self.assertEqual(full.local_edges, fast.local_edges)
        self.assertEqual(full._active_prior, fast._active_prior)
        self.assertEqual(full.llm_counter.call_count, fast.llm_counter.call_count)

    def test_original_files_protected(self):
        self.assertEqual(study.base.preserved()['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
