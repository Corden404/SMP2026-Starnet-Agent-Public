"""Round11 public-information, commuting, state-isolation and transport regressions."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workspace'))
import round11_experiment as study
import round11_api as api
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv
from round4_seeds import generate


def action(kind, node, gain=10.):
    cost = {'shield': 5., 'comm': 2.}[kind]
    return dict(action=kind, target_node_1=node, target_node_2=None,
                prompt_id=1 if kind == 'comm' else None, gain=gain, cost=cost, roi=gain / cost)


class Round11Tests(unittest.TestCase):
    def model(self, variant='F1', index=0, scanned=True):
        module = study.load(variant)
        raw = FiniteEnv(generate(study.specs('dev')[index]))
        logged = LoggedEnv(raw)
        model = study.initialize(module, logged)
        if scanned:
            for _ in range(44):
                self.assertEqual(model.step(), 0)
            self.assertFalse(model._should_scan(raw.get_remaining_budget()))
        return module, raw, logged, model

    def test_exact_s44_and_nine_file_boundary(self):
        import zipfile
        with zipfile.ZipFile(study.ARCHIVE) as z:
            self.assertEqual(study.source_files('S44'), {n: z.read(n) for n in z.namelist()})
        for v in ('R0', 'F1'):
            files = study.source_files(v)
            self.assertEqual(len(files), 9)
            self.assertTrue(files['starnet_model.py'].startswith(study.source_files('S44')['starnet_model.py']))
            for name in files:
                if name != 'starnet_model.py':
                    self.assertEqual(files[name], study.source_files('S44')[name])
            for forbidden in (b'round11_experiment', b'round11_api', b'true_edges', b'true_w'):
                self.assertNotIn(forbidden, files['starnet_model.py'])

    def test_same_scans_and_unknown_targets_ineligible(self):
        views = []
        for v in study.VARIANTS:
            module, raw, logged, m = self.model(v)
            self.assertEqual(len(raw._scanned), 44)
            self.assertEqual(raw.get_remaining_budget(), 78.)
            self.assertIsNone(m._next_scan_target())
            state = m.completed_state()
            self.assertEqual(len(state[0]), 50)
            self.assertTrue(all('r' not in d for d in state[0].values()))
            unknown = set(state[0]) - set(m.local_nodes)
            engine = module.RestrictedRollout(*state, module.SEARCH, 70, eligible=m.local_nodes)
            self.assertTrue(all(a['target_node_1'] not in unknown for a in engine.candidates(engine.initial, 78.)))
            views.append((m.local_nodes, m.local_edges, m.scan_queue, m.llm_counter.call_count))
        self.assertEqual(views[0], views[1]); self.assertEqual(views[0], views[2])

    def test_distinct_shield_and_first_comm_commute_in_six_graph_families(self):
        for index in range(6):
            _, ra, _, ma = self.model('S44', index)
            _, rb, _, mb = self.model('S44', index)
            a, b = list(ma.local_nodes)[:2]
            def perform(model, raw, kind, node):
                chosen = action(kind, node)
                before = dict(model.local_nodes[node])
                model._execute_planned_action(chosen, raw.get_remaining_budget(), before)
            perform(ma, ra, 'shield', a); perform(ma, ra, 'comm', b)
            perform(mb, rb, 'comm', b); perform(mb, rb, 'shield', a)
            self.assertEqual(ra._nodes, rb._nodes)
            self.assertEqual(ra._edges, rb._edges)
            self.assertEqual(ra.get_remaining_budget(), rb.get_remaining_budget())
            self.assertEqual(ma.comm_effects, mb.comm_effects)
            self.assertEqual(ma.persona_observations, mb.persona_observations)
            self.assertEqual(ma.local_nodes, mb.local_nodes)
            self.assertEqual(ma.dead_nodes, mb.dead_nodes)
            self.assertEqual(ma.operation_count, mb.operation_count)
            self.assertEqual(ra.evaluate(), rb.evaluate())

    def test_shadow_uses_no_env_and_isolates_mutable_state(self):
        module, raw, logged, m = self.model()
        a, b = list(m.local_nodes)[:2]
        signature = m._feedback_signature(78.)
        original_actions = copy.deepcopy(logged.actions)
        def shadow_query(shadow, budget):
            self.assertIsNone(shadow.env)
            self.assertEqual(budget, 73.)
            self.assertNotIn(a, shadow.local_nodes)
            shadow.local_nodes[b]['w'] += 999
            shadow.comm_effects[999] = 1.
            shadow.persona_observations['test'] = {999: 1.}
            shadow.search_records.append({'plans': [{'search_completed': True, 'search_truncated': False}]})
            return [action('comm', b)]
        with patch.object(module._FeedbackBase, '_learned_candidates', shadow_query):
            probe, reason = m._feedback_shadow_next(action('shield', a), 78.)
        self.assertEqual(reason, 'eligible'); self.assertEqual(probe['target_node_1'], b)
        self.assertEqual(m._feedback_signature(78.), signature)
        self.assertEqual(logged.actions, original_actions)

    def propose(self, variant='F1'):
        module, raw, logged, m = self.model(variant)
        a, b = list(m.local_nodes)[:2]
        with patch.object(module._FeedbackBase, '_learned_candidates', return_value=[action('shield', a)]), \
             patch.object(m, '_feedback_shadow_next', return_value=(action('comm', b), 'eligible')):
            chosen = m._learned_candidates(78.)[0]
        self.assertEqual(chosen['action'], 'comm')
        self.assertEqual(raw.get_remaining_budget(), 78.)
        self.assertIsNotNone(m._feedback_pending)
        return module, raw, logged, m, a, b, chosen

    def test_control_commits_after_actual_feedback(self):
        module, raw, logged, m, a, b, chosen = self.propose('R0')
        m._execute_planned_action(chosen, 78., dict(m.local_nodes[b]))
        self.assertTrue(m._feedback_pending['ready'])
        response = m.comm_effects[b]
        resolved = m._learned_candidates(76.)[0]
        self.assertEqual((resolved['action'], resolved['target_node_1']), ('shield', a))
        self.assertEqual(m.feedback_records[-1]['observed_first_effect'], response)
        self.assertIsNone(m._feedback_pending)

    def test_adaptive_comparison_occurs_only_after_real_receipt(self):
        module, raw, logged, m, a, b, chosen = self.propose()
        self.assertNotIn(b, m.comm_effects)
        m._execute_planned_action(chosen, 78., dict(m.local_nodes[b]))
        alternative = action('comm', b)
        def compare(state, budget, original, other):
            self.assertEqual(state[2][b], m.comm_effects[b])
            self.assertEqual(budget, 76.)
            self.assertEqual(original['target_node_1'], a)
            return other, {'comparison': 'complete', 'replaced': True}
        with patch.object(module._FeedbackBase, '_learned_candidates', return_value=[alternative]), \
             patch.object(m, '_feedback_compare', side_effect=compare) as check:
            resolved = m._learned_candidates(76.)[0]
        self.assertEqual(resolved, alternative); self.assertEqual(check.call_count, 1)
        self.assertIsNone(m._feedback_pending)

    def test_failed_probe_clears_pending(self):
        _, raw, _, m, a, b, chosen = self.propose()
        with patch.object(m.env, 'communicate', return_value={'status': 'failed'}):
            m._execute_planned_action(chosen, 78., dict(m.local_nodes[b]))
        self.assertIsNone(m._feedback_pending)
        self.assertNotIn(b, m.comm_effects)
        self.assertEqual(raw.get_remaining_budget(), 78.)

    def test_modified_state_invalidates_pending(self):
        module, _, _, m, a, b, chosen = self.propose('R0')
        m._execute_planned_action(chosen, 78., dict(m.local_nodes[b]))
        m.local_nodes[a]['w'] += 1.
        with patch.object(module._FeedbackBase, '_learned_candidates', return_value=[]):
            self.assertEqual(m._learned_candidates(76.), [])
        self.assertEqual(m.feedback_records[-1]['event'], 'pending_invalid')

    def test_comparison_timeout_keeps_original(self):
        _, _, _, m = self.model()
        a, b = list(m.local_nodes)[:2]
        m.feedback_seconds = 0.
        original = action('shield', a)
        chosen, evidence = m._feedback_compare(m.completed_state(), 78., original, action('comm', b))
        self.assertEqual(chosen, original)
        self.assertEqual(evidence['comparison'], 'timeout_keep_original')

    def test_insufficient_budget_or_slots_does_not_probe(self):
        module, _, _, m = self.model()
        a = next(iter(m.local_nodes))
        with patch.object(module._FeedbackBase, '_learned_candidates', return_value=[action('shield', a)]), \
             patch.object(m, '_feedback_shadow_next') as shadow:
            m._learned_candidates(6.)
            m.llm_counter.call_count = m.llm_counter.safe_limit - 1
            m._learned_candidates(78.)
        shadow.assert_not_called()

    def test_other_sizes_bypass_new_policy(self):
        module = study.load('F1')
        for n in (4, 100):
            spec = dict(study.specs('dev')[0], n=n)
            model = study.initialize(module, FiniteEnv(generate(spec)), n=n)
            self.assertEqual(model.scan_target, n)
            with patch.object(module._FeedbackBase, '_learned_candidates', return_value=[]) as parent:
                self.assertEqual(model._learned_candidates(50.), [])
                self.assertEqual(parent.call_count, 1)
            self.assertEqual(model.feedback_records, [])

    def test_real_casevo_and_fast_scan_entry_match(self):
        for v in ('R0', 'F1'):
            module, raw, _, fast = self.model(v)
            other = FiniteEnv(generate(study.specs('dev')[0]))
            with contextlib.redirect_stdout(io.StringIO()):
                full = study.initialize(module, other, mode='integration')
                for _ in range(44):
                    full.step()
            self.assertEqual(full.local_nodes, fast.local_nodes)
            self.assertEqual(full.local_edges, fast.local_edges)
            self.assertEqual(full.llm_counter.call_count, fast.llm_counter.call_count)
            self.assertEqual(full.feedback_records, [])

    def test_api_exact_order_and_model_specific_thinking(self):
        self.assertEqual(api.MODELS, ('glm-4-plus', 'glm-4-air', 'glm-4.6v'))
        self.assertNotIn('thinking', api.config('glm-4-air'))
        self.assertEqual(api.config('glm-4.6v')['thinking'], {'type': 'disabled'})
        with self.assertRaises(ValueError):
            api.config('glm-4.5-air')

    def test_credit_failure_one_request_no_secret_in_journal(self):
        response = SimpleNamespace(status_code=429, text='{"error":{"code":"1113"}}',
                                   json=lambda: {'error': {'code': '1113'}})
        with tempfile.TemporaryDirectory() as directory, patch.object(api, 'key', return_value='FAKE-PRIVATE-KEY'), \
             patch.object(api.requests, 'post', return_value=response) as post:
            llm = api.JournalLLM('glm-4-air', Path(directory) / 'requests.jsonl')
            with self.assertRaises(api.APICreditExhausted):
                llm.send_message('test')
            self.assertEqual(post.call_count, 1)
            self.assertEqual(llm.http_request_count, 1)
            journal = (Path(directory) / 'requests.jsonl').read_text(encoding='utf-8')
            self.assertNotIn('FAKE-PRIVATE-KEY', journal)
            self.assertNotIn('Authorization', journal)

    def test_originals_unchanged(self):
        self.assertEqual(study.preserved()['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
