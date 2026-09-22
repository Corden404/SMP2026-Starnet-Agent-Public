"""Read-only prefix, diagnostic isolation, resource and information-boundary tests."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round14_branches as b

class BranchTests(unittest.TestCase):
    def row(self):return b.read(b.SOURCE_DIRS['scan']/'r12-dev-005-V2.json')
    def test_all_states_selected_without_outcome_filter(self):
        scans=b.entries('scan');feedback=b.entries('feedback')
        self.assertEqual(len(scans),32);self.assertEqual(len(feedback),73)
        self.assertEqual(sum(e['event'].get('reason')=='purchase' for e in scans),3)
        self.assertEqual(sum('pair' not in e['event'] for e in scans),4)
    def test_restore_public_prefix_exact_and_freezes_initial_mean(self):
        row=self.row();module,raw,logged,model=b.restore(row,67)
        self.assertEqual(len(logged.actions),67);self.assertEqual(model.operation_count,67)
        self.assertEqual(raw.get_remaining_budget(),14.)
        initial=[a['result']['w'] for a in row['actions'][:44]]
        self.assertAlmostEqual(model.unseen_w,sum(initial)/44)
        self.assertEqual(model.llm_counter.call_count,sum(c['operation']<67 for c in row['chains']))
        self.assertFalse(hasattr(model.env,'_nodes'))
        self.assertTrue(all('r' not in d for d in model.local_nodes.values()))
    def test_scan_info_is_public_and_restriction_blocks_new_targets(self):
        module,raw,logged,model=b.restore(self.row(),67);eligible=set(model.local_nodes)
        for node in (46,33):b.force(model,raw,model._action_dict('scan',node),reserve=False)
        self.assertEqual(raw.get_remaining_budget(),13.)
        self.assertFalse(model._should_scan(13.));self.assertIsNone(model._next_scan_target())
        b.restrict_candidates(module,model,eligible)
        choices=model._learned_candidates(13.)
        self.assertTrue(all(a['target_node_1'] in eligible for a in choices))
        self.assertTrue(all('r' not in d for d in model.completed_state()[0].values()))
    def test_restore_does_not_reuse_future_receipts(self):
        row=self.row();mod,raw,logged,m=b.restore(row,67)
        self.assertNotIn(46,m.local_nodes);self.assertNotIn(33,m.local_nodes)
        self.assertEqual(len(raw._scanned),44)
    def test_force_real_feedback_only_after_env_call(self):
        row=b.read(b.SOURCE_DIRS['feedback']/'r11-dev-002-F1.json')
        mod,raw,logged,m=b.restore(row,70,'F1')
        self.assertNotIn(9,m.comm_effects)
        b.force(m,raw,m._action_dict('comm',9))
        self.assertAlmostEqual(m.comm_effects[9],19.18575)
        self.assertEqual(raw.get_remaining_budget(),6.)
    def test_explicit_charge_is_not_a_fabricated_receipt(self):
        mod,raw,logged,m=b.restore(self.row(),67)
        before=b.public_state(m,raw);actions=copy.deepcopy(logged.actions)
        raw._budget-=1;m.operation_count+=2;m.llm_counter._reserve_call()
        after=b.public_state(m,raw)
        self.assertEqual(after['nodes'],before['nodes']);self.assertEqual(after['edges'],before['edges'])
        self.assertEqual(after['budget'],before['budget']-1)
        self.assertEqual(after['operations'],before['operations']+2)
        self.assertEqual(logged.actions,actions)
    def test_complete_group_entry_routes_all_four_branches(self):
        import tempfile
        entry=next(e for e in b.entries('scan') if 'pair' in e['event'])
        protocol={'entries':{'scan':[entry]},'code':{'test':'entry-routing'}}
        with tempfile.TemporaryDirectory() as d,patch.object(b,'OUT',Path(d)),patch.object(b,'check',return_value=protocol),patch.object(b,'scan_branch',return_value={'final_score':0.}) as branch:
            result=b.run_group('scan',0)
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(branch.call_count,4)
            stored=b.read(Path(d)/'runs/scan'/(entry['id']+'.json'))
            self.assertEqual(set(stored['branches']),{'A','B','C','D'})
            self.assertEqual(stored['differences']['total'],0.)

    def test_originals_unchanged(self):self.assertEqual(b.r13.preserved()['status'],'PASS')

if __name__=='__main__':unittest.main()
