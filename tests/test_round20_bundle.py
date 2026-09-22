import copy,sys,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round20_bundle_diagnose as d
import round19_common as h

class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.m=h.load('Numeric','r20-test')
    def public(self):
        old=Path('agent_notes/experiments/round19/diagnostic/states/dev-000-budget20.json')
        r=d.s.read(old);e=r['identity']['entry'];row=d.s.read(d.s.ROOT/e['source_path'])
        return row,r
    def test_inventory_exact(self):
        xs=d.inventory();self.assertEqual(len(xs),132);self.assertEqual(len({x['id'] for x in xs}),132)
    def test_candidate_cap_and_structural_only(self):
        row,r=self.public();state,eligible,ctx=h.public_at(row,r['snapshot'])
        xs=d.bundle_candidates(self.m,state,r['snapshot']['budget'],eligible)
        self.assertLessEqual(len(xs),7)
        for _,prefix in xs:
            self.assertLessEqual(len(prefix),2);self.assertTrue(all(a['action'] in ('cut','shield') for a in prefix))
            self.assertTrue(all(a['gain']>.25 for a in prefix))
            self.assertTrue(all(a['target_node_1'] in eligible and (a['action']!='cut' or a['target_node_2'] in eligible) for a in prefix))
    def test_structural_apply_does_not_mutate(self):
        row,r=self.public();state,eligible,ctx=h.public_at(row,r['snapshot']);before=copy.deepcopy(state)
        xs=d.bundle_candidates(self.m,state,r['snapshot']['budget'],eligible)
        if xs:d.structural_apply(self.m,state,xs[0][1][0],eligible)
        self.assertEqual(state,before)
    def test_zero_candidate_forecast_no_selection(self):
        row,r=self.public();state,eligible,ctx=h.public_at(row,r['snapshot'])
        with h.old.c.offline():
            out=d.forecast(self.m,state,r['snapshot']['budget'],eligible,ctx,r['incumbent'],[],12.)
        self.assertTrue(out['all_complete']);self.assertIsNone(out['selected'])
    def test_same_single_incumbent_scores_zero_when_structural(self):
        row,r=self.public();state,eligible,ctx=h.public_at(row,r['snapshot'])
        inc=r['incumbent']
        if inc['action'] not in ('cut','shield'):self.skipTest('fixture incumbent not structural')
        with h.old.c.offline():
            out=d.forecast(self.m,state,r['snapshot']['budget'],eligible,ctx,inc,[('same',[inc])],12.)
        self.assertAlmostEqual(out['plans'][0]['mean_delta'],0.,places=8)
if __name__=='__main__':unittest.main()
