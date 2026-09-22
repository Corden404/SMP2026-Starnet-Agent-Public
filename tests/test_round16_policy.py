"""PC2 information boundaries, equal-world controls, scorer and entry tests."""
import copy
from pathlib import Path
import random
import sys
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round16_common as c
import round16_experiment as e

class PC2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=e.load('PC2')
    def state(self):
        return ({1:dict(w=4.,persona='和平',comm_left=3),2:dict(w=-4.,persona='暴力',comm_left=3),
                 3:dict(w=1.,persona='中立',comm_left=0)}, {(1,2),(2,3)}, {}, {})
    def comm(self,i):
        return dict(action='comm',target_node_1=i,target_node_2=None,prompt_id=1,cost=2.,gain=1.,roi=.5)
    def test_numpy_private_score_matches_independent_forward(self):
        from mechanism_round5 import score as direct
        rng=random.Random(1699)
        for n in (1,4,13,50,100):
            for density in (.02,.15,.6):
                nodes={i:dict(w=rng.uniform(-50,50),persona='中立',comm_left=3) for i in range(1,n+1)}
                edges={(i,j) for i in nodes for j in nodes if i<j and rng.random()<density}
                got=self.m.pc2_score(nodes,edges)[0]
                expected=direct(dict(nodes=[dict(id=i,**d) for i,d in nodes.items()],edges=list(edges)))
                self.assertAlmostEqual(got,expected,places=8)
    def test_numpy_missing_falls_back_to_original_score(self):
        self.m.pc2_coefficients.cache_clear()
        with patch.object(self.m,'_pc_np',None):
            s=self.state();self.assertAlmostEqual(self.m.pc2_score(*s[:2])[0],self.m.score(*s[:2])[0],places=10)
        self.m.pc2_coefficients.cache_clear()
    def test_private_candidates_match_original(self):
        state=self.state();original=self.m.RestrictedRollout(*state,self.m.SEARCH,8,eligible={1,2})
        private=self.m.PC2Rollout(*state,self.m.SEARCH,8,eligible={1,2})
        for budget in (2,3,5,10):
            a=original.candidates(state,budget);b=private.candidates(state,budget)
            self.assertEqual([original.key(x) for x in a],[private.key(x) for x in b])
            for x,y in zip(a,b):self.assertAlmostEqual(x['gain'],y['gain'],places=8)
    def test_same_action_same_world_values_zero(self):
        state=self.state();first=self.comm(1)
        chosen,result=self.m.pc2_compare(state,4,2,{1,2},first,first,seconds=3.,depth=2)
        self.assertEqual(result['predicted_delta'],0.)
        self.assertFalse(result['replaced']);self.assertEqual(result['completed_worlds'],3)
    def test_simulation_does_not_mutate_public_state(self):
        state=self.state();before=copy.deepcopy(state)
        self.m.pc2_compare(state,4,2,{1,2},self.comm(1),self.comm(2),seconds=3.,depth=2)
        self.assertEqual(state,before)
    def test_response_is_revealed_only_after_action(self):
        state=self.state();world={1:80.,2:1.,3:1.};seen=[]
        def selector(belief,budget,slots,eligible,deadline,replan=True):
            seen.append(copy.deepcopy(belief));return None
        with patch.object(self.m,'pc2_next',side_effect=selector):
            self.m.pc2_continue(state,6,3,{1,2},self.comm(1),world,time.monotonic()+3.)
        self.assertEqual(state[2],{})
        self.assertEqual(seen[0][2],{1:80.})
        self.assertNotIn(2,seen[0][2]);self.assertNotIn('r',seen[0][0][2])
    def test_measured_response_fixed_all_worlds(self):
        s=self.state();s[2][1]=7.
        worlds=self.m.pc2_worlds(s)
        self.assertTrue(all(w[1]==7. for w in worlds))
    def test_hidden_input_rejected(self):
        s=self.state();s[0][1]['r']=2.
        with self.assertRaises(ValueError):self.m.pc2_worlds(s)
    def test_timeout_keeps_original(self):
        a=self.comm(1);chosen,result=self.m.pc2_compare(self.state(),4,2,{1,2},a,self.comm(2),seconds=0.)
        self.assertEqual(chosen,a);self.assertEqual(result['completed_worlds'],0)
    def test_partial_worlds_never_change_action(self):
        a=self.comm(1)
        with patch.object(self.m,'pc2_continue',side_effect=[{'score':0.},{'score':100.},TimeoutError()]):
            chosen,result=self.m.pc2_compare(self.state(),4,2,{1,2},a,self.comm(2))
        self.assertEqual(chosen,a);self.assertEqual(result['completed_worlds'],1)
        self.assertEqual(result['reason'],'incomplete_keep_original')
    def test_other_sizes_bypass(self):
        from experiment_round5 import FiniteEnv
        for n in (4,100):
            spec=dict(c.specs('integration')[0],n=n)
            with c.offline():
                model=c.old.initialize(self.m,FiniteEnv(c.generate(spec)),n=n,mode='integration')
                self.assertEqual(model.scan_target,n)
                for _ in range(4):model.step()
            self.assertEqual(model.pc2_records,[])
    def test_gc2_only_changes_depth(self):
        self.assertEqual(e.load('GC2').ParticipantSquadModel.pc2_depth,0)
        self.assertEqual(self.m.ParticipantSquadModel.pc2_depth,2)
        for name,data in e.source_files('S44').items():
            if name!='starnet_model.py':self.assertEqual(data,e.source_files('PC2')[name])
    def test_eligible_targets_never_include_unknown(self):
        state=self.state();private=self.m.PC2Rollout(*state,self.m.SEARCH,8,eligible={1})
        self.assertTrue(all(a['target_node_1']==1 and a['action']!='cut' for a in private.candidates(state,10)))

if __name__=='__main__':unittest.main()
