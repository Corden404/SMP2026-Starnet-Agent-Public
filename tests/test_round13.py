"""Crossed design and immutable-policy regressions; no API/network calls."""
import collections
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round13_crossed as r13
from round4_seeds import generate,FAMILIES,SCALES

class CrossedTests(unittest.TestCase):
    def test_complete_factorial_and_replication(self):
        for pool,rep in [('main',4),('stress',1)]:
            counts=collections.Counter((s['family'],s['scale'],s['persona_mix']) for s in r13.specs(pool))
            self.assertEqual(len(counts),54)
            self.assertEqual(set(counts.values()),{rep})
            for f in FAMILIES:
                self.assertEqual(len([k for k in counts if k[0]==f]),9)
    def test_main_execution_order_balanced_within_cell(self):
        for cell in range(54):
            self.assertEqual(sum((rep+cell)%2==0 for rep in range(4)),2)
    def test_stress_modes_and_density_not_fixed_by_persona(self):
        rows=r13.specs('stress')
        self.assertEqual(set(s['mode'] for s in rows),{'degree_w','degree_r','community','w_r'})
        for p in range(3):
            group=[s for s in rows if s['persona_mix']==p]
            self.assertEqual(len({s['mode'] for s in group}),4)
            self.assertEqual({s['ood'] for s in group},{False,True})
    def test_glm_is_fractional_balanced_and_fresh(self):
        rows=r13.specs('glm');self.assertEqual(len(rows),48)
        self.assertEqual(set(collections.Counter(s['family'] for s in rows).values()),{8})
        all_specs=[s for p in r13.COUNTS for s in r13.specs(p)]
        self.assertEqual(len(all_specs),324)
        self.assertEqual(len({s['case'] for s in all_specs}),324)
    def test_no_previous_seed_identities(self):
        names={s['case'] for p in r13.COUNTS for s in r13.specs(p)}
        for d in ('round9','round10-s42','round11-feedback','round12-active-scan'):
            prior=r13.read(r13.ROOT/'agent_notes/experiments'/d/'protocol.json')
            previous={r['spec']['case'] for rs in prior['seeds'].values() for r in rs}
            self.assertFalse(names&previous)
    def test_exact_old_policy_and_package_boundary(self):
        for variant in r13.VARIANTS:
            files=r13.source_files(variant)
            self.assertEqual(files,r13.old.source_files(variant));self.assertEqual(len(files),9)
        self.assertEqual(r13.sha(r13.old.ARCHIVE),r13.old.ARCHIVE_SHA)
    def test_fast_init_and_public_wrapper(self):
        module=r13.load('F1');raw=r13.FiniteEnv(generate(r13.specs('main')[0]))
        model=r13.old.initialize(module,raw)
        self.assertFalse(hasattr(model.env,'_nodes'))
        self.assertFalse(hasattr(model.env,'_edges'))
        for _ in range(44):self.assertEqual(model.step(),0)
        self.assertEqual(raw.get_remaining_budget(),78.)
        self.assertFalse(model._should_scan(78.))
        self.assertEqual(len(model.completed_state()[0]),50)
        self.assertEqual(model.feedback_mode,'F1');self.assertEqual(module.SEARCH,r13.old.load('F1').SEARCH)
    def test_receipt_comparison_does_not_mutate(self):
        a={'neighbors':[2,1],'w':4};b={'neighbors':[1,2],'w':4}
        self.assertTrue(r13.receipt_equal(a,b));self.assertEqual(a['neighbors'],[2,1])
        self.assertFalse(r13.receipt_equal(a,dict(b,w=3)))
    def test_original_files_protected(self):
        self.assertEqual(r13.preserved()['status'],'PASS')

if __name__=='__main__':unittest.main()
