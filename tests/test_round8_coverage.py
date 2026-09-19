import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
from round8_common import prepare_baseline
prepare_baseline()
from round8_coverage import M, CoverageRollout, ROICoverageRollout

class CoverageTests(unittest.TestCase):
    def state(self):
        weights={1:30.,2:-40.,3:15.,4:-10.,5:50.,6:5.}
        nodes={i:dict(w=w,persona='中立',comm_left=3) for i,w in weights.items()}
        return nodes,{(1,2),(2,3),(3,4),(4,5),(5,6),(2,5)},{i:12. for i in nodes},{}

    def test_completed_comparison_never_selects_lower_predicted_score(self):
        for budget in (5,8,14,22):
            for cls in (CoverageRollout,ROICoverageRollout):
                state=self.state();before=copy.deepcopy(state)
                opts=dict(M.SEARCH,rollout_seconds=10.)
                base=M.RestrictedRollout(*state,opts,20,eligible=state[0]).choose(budget)
                selected=cls(*state,opts,20,eligible=state[0]).choose(budget)
                self.assertGreaterEqual(selected[0]['predicted_final']+1e-8,base[0]['predicted_final'])
                self.assertEqual(state,before)
                self.assertLessEqual(selected[0]['first']['cost'],budget)

    def test_zero_budget_search_keeps_original_fallback(self):
        state=self.state();opts=dict(M.SEARCH,rollout_seconds=0.)
        base=M.RestrictedRollout(*state,opts,20,eligible=state[0]).choose(10)
        new=CoverageRollout(*state,opts,20,eligible=state[0]).choose(10)
        self.assertEqual(new,base)

    def test_unknown_vertices_never_become_extra_targets(self):
        state=self.state();state[0][6]['comm_left']=0
        engine=CoverageRollout(*state,M.SEARCH,20,eligible={1,2,3,4,5})
        for a in engine.candidates(state,20,True):
            self.assertNotEqual(a['target_node_1'],6);self.assertNotEqual(a['target_node_2'],6)
        for p in engine.choose(20):
            self.assertNotEqual(p['first']['target_node_1'],6)
            self.assertNotEqual(p['first']['target_node_2'],6)

if __name__=='__main__':unittest.main()
