"""Offline follow-up controls. No API credentials or paid calls."""
import collections
import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workspace'))
import round10_s42 as study
from round4_seeds import generate
from experiment_round3 import fast_model, TopChoiceLLM
from experiment_round5 import FiniteEnv, PublicEnv


class S42FollowupTests(unittest.TestCase):
    def test_fresh_complete_cohorts(self):
        old = study.read(study.OLD / 'protocol.json')
        old_ids = {r['spec']['case'] for rows in old['seeds'].values() for r in rows}
        new = [s for p in study.COUNTS for s in study.specs(p)]
        self.assertEqual(len(new), 150)
        self.assertEqual(len({s['case'] for s in new}), 150)
        self.assertFalse(old_ids.intersection(s['case'] for s in new))
        for p, n in study.COUNTS.items():
            self.assertEqual(set(collections.Counter(s['family'] for s in study.specs(p)).values()), {n // 6})
        self.assertEqual({s['mode'] for s in study.specs('stress')}, {'degree_w','degree_r','community','w_r'})

    def test_only_scan_target_changes(self):
        b = study.common.source_files('S44')
        c = study.common.source_files('S42')
        self.assertEqual({k for k in b if b[k] != c[k]}, {'starnet_model.py'})
        self.assertEqual(c['starnet_model.py'], b['starnet_model.py'] + b'\n\nclass ParticipantSquadModel(ParticipantSquadModel):\n    scan_target = 42\n')

    def test_gate_requires_complete_and_non_fragile_positive(self):
        c = dict(n=96, mean=2., without_best_two=1.)
        s = dict(n=24, mean=0.)
        self.assertTrue(study.local_eligible(c,s))
        for k,v in (('n',95),('mean',0.),('without_best_two',-.01)):
            self.assertFalse(study.local_eligible(dict(c, **{k:v}),s))
        self.assertFalse(study.local_eligible(c,dict(s,mean=-.01)))

    def test_pair_order_balanced_within_families(self):
        for pool in ('confirm','stress','glm'):
            for family in range(6):
                first = [(i // 6 + i % 6) % 2 for i in range(study.COUNTS[pool]) if i % 6 == family]
                self.assertEqual(first.count(0), first.count(1))

    def test_scans_and_persistent_partial_graph(self):
        # Use the existing frozen module without changing round9 module paths.
        module = study.common.load('S42')
        env = FiniteEnv(generate(study.specs('confirm')[0]))
        model = study.runner.initialize(fast_model(PublicEnv(env), copy.deepcopy(study.common.people()), module.ParticipantSquadModel))
        model.llm_counter = module.CountingLLM(TopChoiceLLM(),model.llm_limit)
        for _ in range(42): self.assertEqual(model.step(), 0)
        self.assertEqual(len(env._scanned),42)
        self.assertEqual(env.get_remaining_budget(),79.)
        self.assertFalse(model._should_scan(79.))
        self.assertEqual(len(model.completed_state()[0]),50)
        unknown = set(range(1,51)) - set(model.local_nodes)
        self.assertEqual(len(unknown),8)
        self.assertTrue(all(model.completed_state()[0][i]['comm_left'] == 0 for i in unknown))
        first = min(model.local_nodes)
        model._dispatch_env_action('shield',first,None,None,79.)
        self.assertFalse(model._should_scan(74.))
        self.assertEqual(len(model.completed_state()[0]),49)

    def test_other_sizes_keep_original_scan_policy(self):
        module = study.common.load('S42')
        for n in (4,100):
            spec = dict(study.specs('accept')[0],n=n)
            people = copy.deepcopy(study.common.people());people[0]['runtime']['node_count'] = n
            with contextlib.redirect_stdout(io.StringIO()):
                model = module.ParticipantSquadModel(PublicEnv(FiniteEnv(generate(spec))),people,TopChoiceLLM())
            self.assertEqual(model.scan_target,n)


if __name__ == '__main__': unittest.main()
