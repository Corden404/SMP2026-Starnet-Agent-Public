"""Actual CaseVO full games and fixed-window PC2 scheduling controls."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round16_common as c
import round16_experiment as e
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv

class Round16AcceptanceTests(unittest.TestCase):
    def test_complete_actual_casevo_4_50_100_and_independent_score(self):
        module=e.load('PC2')
        for n in (4,50,100):
            seed=dict(global_setting=dict(max_budget=2*n,max_api_calls=250 if n==100 else 120),
                nodes=[dict(id=i,w=5.,r=1.,persona='和平',comm_left=3) for i in range(1,n+1)],
                edges=[],prompts={'1':15.,'2':10.,'3':-5.},original_total=5*n)
            raw=FiniteEnv(seed);logged=LoggedEnv(raw)
            with c.offline():
                model=c.old.initialize(module,logged,n=n,mode='integration')
                row=c.finish(raw,logged,model,n=n)
            scans=44 if n==50 else n
            communications=int((2*n-.5*scans)//2)
            self.assertEqual(len(raw._scanned),scans)
            self.assertAlmostEqual(row['final_score'],5*n+15*communications,places=8)
            self.assertEqual(row['operations'],scans+communications)
            self.assertLess(row['logical_calls'],250 if n==100 else 120)
            if n!=50:self.assertEqual(model.pc2_records,[])
    def make_model(self):
        m=e.load('PC2');model=m.ParticipantSquadModel.__new__(m.ParticipantSquadModel)
        model.node_count=50;model.operation_count=44;model.operation_limit=120
        model.llm_counter=types.SimpleNamespace(safe_limit=118,call_count=5)
        model.local_nodes={1:dict(w=1.,persona='和平',comm_left=3),2:dict(w=1.,persona='和平',comm_left=3)}
        model.completed_state=lambda:(model.local_nodes,set(),{},{});model._pc2_init()
        a=dict(action='comm',target_node_1=1,target_node_2=None,prompt_id=1,cost=2.,gain=1.,roi=.5)
        b=dict(a,target_node_1=2)
        return m,model,a,b
    def test_exactly_four_fixed_windows_no_extra_environment_calls(self):
        m,model,a,b=self.make_model()
        with patch.object(m._PC2Base,'_learned_candidates',return_value=[a]),patch.object(m,'pc2_pair',return_value=[a,b]),patch.object(m,'pc2_compare',return_value=(b,dict(reason='replaced',replaced=True))) as compare:
            for index,budget in enumerate((78,77,40,39,20,19,10,9)):
                model.operation_count=44+index
                model._learned_candidates(budget)
            self.assertEqual(compare.call_count,4)
            self.assertEqual(model._pc2_seen,{'first','budget40','budget20','budget10'})
            self.assertEqual(len(model.pc2_records),4)
        self.assertFalse(hasattr(model,'env'))
    def test_one_original_prefix_abstains_without_compare(self):
        m,model,a,b=self.make_model()
        with patch.object(m._PC2Base,'_learned_candidates',return_value=[a]),patch.object(m,'pc2_pair',return_value=[a]),patch.object(m,'pc2_compare') as compare:
            self.assertEqual(model._learned_candidates(78),[a]);compare.assert_not_called()
            self.assertEqual(model.pc2_records[0]['reason'],'no_second_original_prefix')
    def test_out_of_pair_original_preserved(self):
        m,model,a,b=self.make_model();other=dict(a,target_node_1=3)
        with patch.object(m._PC2Base,'_learned_candidates',return_value=[other]),patch.object(m,'pc2_pair',return_value=[a,b]),patch.object(m,'pc2_compare') as compare:
            self.assertEqual(model._learned_candidates(78),[other]);compare.assert_not_called()

if __name__=='__main__':unittest.main()
