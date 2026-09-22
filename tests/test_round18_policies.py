"""Exact arithmetic, private-cache isolation, public information and fallback tests."""
import copy
from pathlib import Path
import random
import sys
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round18_common as h

class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=h.load('PC2','policy-test');cls.cut=h.load('Cut1','policy-test');cls.fast=h.load('Fast','policy-test')
    def state(self,n=4,seed=1801):
        rng=random.Random(seed);ids=list(range(1,n+1));rng.shuffle(ids)
        nodes={i:dict(w=rng.uniform(-50,30),persona=('和平','中立','暴力')[i%3],comm_left=3-i%4) for i in ids}
        edges={(i,j) for i in ids for j in ids if i<j and rng.random()<.14}
        return nodes,edges,{},{}
    def comm(self,i=1):
        return dict(action='comm',target_node_1=i,target_node_2=None,prompt_id=1,cost=2.,gain=1.,roi=.5)
    def test_only_model_extension_changes(self):
        base=h.source_files('PC2')
        for v in ('Cut1','Fast'):
            files=h.source_files(v)
            self.assertTrue(files['starnet_model.py'].startswith(base['starnet_model.py']))
            self.assertEqual({k:b for k,b in files.items() if k!='starnet_model.py'},
                             {k:b for k,b in base.items() if k!='starnet_model.py'})
    def test_fast_all_candidate_values_and_order_exact(self):
        for n in (1,4,13,50):
            for seed in (1801,1802):
                state=self.state(n,seed);eligible={i for i in state[0] if i%4!=0}
                original=self.fast._R18OriginalPC2Rollout(*state,self.fast.SEARCH,20,eligible=eligible)
                fast=self.fast.R18FastRollout(*state,self.fast.SEARCH,20,eligible=eligible)
                for budget in (2.,3.,4.,5.,10.):
                    for negative in (False,True):
                        a=original.candidates(state,budget,negative);b=fast.candidates(state,budget,negative)
                        self.assertEqual(a,b,(n,seed,budget,negative))
    def test_fast_reuses_topology_but_new_weights_recompute(self):
        self.fast.r18_structure_rows.cache_clear();state=self.state(13)
        engine=self.fast.R18FastRollout(*state,self.fast.SEARCH,20,eligible=state[0])
        first=engine.candidates(state,10,True);cold=self.fast.r18_structure_rows.cache_info()
        state[0][1]['w']+=20
        actual=engine.candidates(state,10,True);warm=self.fast.r18_structure_rows.cache_info()
        expected=self.fast._R18OriginalPC2Rollout(*state,self.fast.SEARCH,20,eligible=state[0]).candidates(state,10,True)
        self.assertEqual(actual,expected);self.assertGreater(warm.hits,cold.hits);self.assertNotEqual(first,actual)
    def test_fast_topology_changes_do_not_use_stale_coefficients(self):
        state=self.state(13);state[1].add((1,2))
        engine=self.fast.R18FastRollout(*state,self.fast.SEARCH,20,eligible=state[0]);engine.candidates(state,10)
        for modified in (self.fast.BudgetRollout.clone(state),self.fast.BudgetRollout.clone(state)):
            if (1,2) in modified[1]:modified[1].remove((1,2))
            expected=self.fast._R18OriginalPC2Rollout(*modified,self.fast.SEARCH,20,eligible=modified[0]).candidates(modified,10,True)
            self.assertEqual(engine.candidates(modified,10,True),expected)
    def test_fast_symmetric_ties_preserve_order(self):
        state=({i:dict(w=0.,persona='和平',comm_left=3) for i in (4,3,2,1)}, {(1,2),(2,3),(3,4),(1,4)}, {},{})
        a=self.fast._R18OriginalPC2Rollout(*state,self.fast.SEARCH,5,eligible=state[0]).candidates(state,10,True)
        b=self.fast.R18FastRollout(*state,self.fast.SEARCH,5,eligible=state[0]).candidates(state,10,True)
        self.assertEqual(a,b)
    def test_cut_best_root_matches_original_sparse(self):
        for seed in (1801,1802,1803):
            state=self.state(13,seed);eligible=set(state[0])-{3}
            candidates=self.cut.RestrictedRollout(*state,self.cut.SEARCH,8,eligible=eligible).candidates(state,10)
            expected=next((a for a in candidates if a['action']=='cut'),None)
            self.assertEqual(self.cut.r18_best_cut(state,10,eligible,time.monotonic()+10),expected)
    def test_cut_private_does_not_warm_old_caches(self):
        module=self.cut;h.clear(module);state=self.state(4);state[0][1]['comm_left']=3
        before=(module.coefficients.cache_info(),module.pc2_coefficients.cache_info())
        module.r18_best_cut(state,10,set(state[0]),time.monotonic()+10)
        module.r18_cut_compare(state,4,2,set(state[0]),self.comm(),self.comm(),seconds=10.)
        self.assertEqual(before,(module.coefficients.cache_info(),module.pc2_coefficients.cache_info()))
    def test_cut_comparison_matches_pc2_complete(self):
        state=self.state(4);state[0][1]['comm_left']=3;state[0][2]['comm_left']=3
        a=self.base.pc2_compare(state,6,3,set(state[0]),self.comm(1),self.comm(2),seconds=30.)
        b=self.cut.r18_cut_compare(state,6,3,set(state[0]),self.comm(1),self.comm(2),seconds=30.)
        self.assertEqual(a[0],b[0]);self.assertEqual(a[1]['worlds'],b[1]['worlds'])
    def test_cut_partial_comparison_keeps_pc2(self):
        first=self.comm()
        with patch.object(self.cut,'r18_cut_continue',side_effect=[{'score':0.},{'score':100.},TimeoutError()]):
            selected,event=self.cut.r18_cut_compare(self.state(),6,3,{1,2},first,self.comm(2))
        self.assertEqual(selected,first);self.assertFalse(event['replaced']);self.assertEqual(event['completed_worlds'],1)
    def test_cut_zero_deadline_and_root_timeout(self):
        first=self.comm();selected,event=self.cut.r18_cut_compare(self.state(),6,3,{1,2},first,self.comm(2),seconds=0.)
        self.assertEqual(selected,first)
        with self.assertRaises(TimeoutError):self.cut.r18_best_cut(self.state(),10,{1,2},time.monotonic()-1)
    def test_cut_simulation_does_not_mutate_belief(self):
        state=self.state();state[0][1]['comm_left']=3;before=copy.deepcopy(state)
        self.cut.r18_cut_compare(state,4,2,set(state[0]),self.comm(),self.comm(),seconds=10.)
        self.assertEqual(state,before)
    def test_cut_rejects_hidden_response(self):
        state=self.state();state[0][1]['r']=2.
        with self.assertRaises(ValueError):self.cut.r18_cut_compare(state,4,2,{1,2},self.comm(),self.comm())
    def test_cut_reveals_only_selected_response(self):
        state=self.state();state[0][1]['comm_left']=3;seen=[]
        def next_action(belief,*args,**kwargs):seen.append(copy.deepcopy(belief));return None
        with patch.object(self.cut,'r18_cut_next',side_effect=next_action):
            self.cut.r18_cut_continue(state,6,3,set(state[0]),self.comm(),{i:100. for i in state[0]},time.monotonic()+10.)
        self.assertEqual(seen[0][2],{1:100.});self.assertEqual(state[2],{})
    def test_numpy_absent_cut_isolated_fallback(self):
        module=self.cut;h.clear(module);state=self.state()
        before=module.coefficients.cache_info()
        with patch.object(module,'_pc_np',None):
            got=module.r18_cut_score(*state[:2])[0]
            expected=module.r18_cut_root_score(*state[:2])
        self.assertEqual(got,expected);self.assertEqual(before,module.coefficients.cache_info());h.clear(module)
    def test_hundred_nodes_exact_private_candidates(self):
        state=self.state(100);state=(state[0],{(i,i+1) for i in range(1,100)},state[2],state[3])
        a=self.fast._R18OriginalPC2Rollout(*state,self.fast.SEARCH,8,eligible={1,2,50,99,100}).candidates(state,6,True)
        b=self.fast.R18FastRollout(*state,self.fast.SEARCH,8,eligible={1,2,50,99,100}).candidates(state,6,True)
        self.assertEqual(a,b)
    def test_original_search_options_unchanged(self):
        self.assertEqual(self.base.SEARCH,self.fast.SEARCH);self.assertEqual(self.base.SEARCH,self.cut.SEARCH)
        self.assertEqual(self.fast.ParticipantSquadModel.pc2_seconds,3.)
        self.assertEqual(self.cut.ParticipantSquadModel.r18_margin,.5)

if __name__=='__main__':unittest.main()
