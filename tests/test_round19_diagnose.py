"""Pre-result diagnostic gate and full inventory checks."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round19_diagnose as d

class DiagnoseTests(unittest.TestCase):
    def test_completed_shadow_with_existing_mode_serializes(self):
        ev=dict(mode='full_pc2',completed_worlds=3,predicted_delta=1.)
        record=d.shadow_record({'case':'a'},'full_pc2',ev)
        self.assertEqual(record['mode'],'full_pc2');self.assertEqual(record['identity'],{'case':'a'})
        self.assertEqual(ev,dict(mode='full_pc2',completed_worlds=3,predicted_delta=1.))

    def test_inventory_exact_primary_and_supplement(self):
        rows=d.inventory()
        self.assertEqual(len(rows),135);self.assertEqual(sum(r['primary'] for r in rows),132)
        self.assertEqual(len({r['case'] for r in rows if r['primary']}),66)
        self.assertEqual(sum(len(r['reuse'] or {}) for r in rows),48)
        self.assertEqual({r['case'] for r in rows if not r['primary']},{'r18-dev-004','r18-dev-028','r18-dev-051'})
    def metrics(self,mae=1.,gain=1.):
        return {'depth2':dict(graph_mae=2.,graph_selected_realized=0.),'full_pc2':dict(graph_mae=mae,graph_selected_realized=gain)}
    def test_improvement_passes(self):self.assertTrue(d.gates(self.metrics(),1.,[])['eligible_ct1'])
    def test_dominated_does_not_pass(self):self.assertFalse(d.gates(self.metrics(mae=3.),1.,[])['eligible_ct1'])
    def test_incomplete_map_not_pass(self):self.assertFalse(d.gates(self.metrics(),.5,[])['eligible_ct1'])
    def test_risk_requires_two_distinct_graphs(self):
        cases=[dict(case='a',actual_delta=-1.),dict(case='a',actual_delta=-2.)]
        self.assertFalse(d.gates(self.metrics(),1.,cases)['eligible_ct1r'])
        cases.append(dict(case='b',actual_delta=2.));self.assertTrue(d.gates(self.metrics(),1.,cases)['eligible_ct1r'])
    def test_no_change_does_not_pass(self):self.assertFalse(d.gates(self.metrics(mae=2.,gain=0.),1.,[])['eligible_ct1'])

if __name__=='__main__':unittest.main()
