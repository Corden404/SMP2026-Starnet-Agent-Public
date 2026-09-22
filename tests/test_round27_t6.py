import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round27_common as h

class TailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=h.load('T6','unit')
    def state(self,known=True):
        nodes={1:dict(w=-8.,persona='中立',comm_left=3),2:dict(w=6.,persona='和平',comm_left=3)}
        return nodes,{(1,2)},{1:7.,2:12.} if known else {},{}
    def action(self,kind='comm',u=2,v=None):
        return dict(action=kind,target_node_1=u,target_node_2=v,prompt_id=1 if kind=='comm' else None,cost=self.m.BudgetRollout.COST[kind])
    def test_archive_and_prompt_identity(self):
        a,b=h.files('PC2'),h.files('T6')
        self.assertEqual(len(b),9);self.assertTrue(b['starnet_model.py'].startswith(a['starnet_model.py']))
        for n in a:
            if n!='starnet_model.py':self.assertEqual(a[n],b[n])
    def test_score_matches_original(self):
        state=self.state()
        self.assertAlmostEqual(self.m.r27_score(*state[:2])[0],self.m.score(*state[:2])[0],places=10)
    def test_private_score_does_not_warm_root(self):
        self.m.coefficients.cache_clear();self.m.pc2_coefficients.cache_clear();state=self.state()
        self.m.r27_score(*state[:2])
        self.assertEqual(self.m.coefficients.cache_info().currsize,0);self.assertEqual(self.m.pc2_coefficients.cache_info().currsize,0)
    def test_known_toy_exact_against_bruteforce(self):
        m=self.m;state=self.state();planner=m.R27Tail({1,2},seconds=60,max_work=100000)
        def brute(st,b,slots):
            best=m.r27_score(*st[:2])[0]
            if b<2 or slots==0:return best
            for a in planner.candidates(st,b):
                ns=planner.apply(st,a,st[2].get(a['target_node_1']))
                best=max(best,brute(ns,b-a['cost'],slots-1))
            return best
        for b in (2,3,4,5,6):
            self.assertAlmostEqual(planner.value(state,b,3)[0],brute(state,b,3),places=8)
    def test_unknown_is_not_preobserved(self):
        state=self.state(False);before=copy.deepcopy(state);planner=self.m.R27Tail({1,2},seconds=60)
        after=planner.apply(state,self.action(),9.)
        self.assertEqual(after[2],{2:9.});self.assertNotIn(1,after[2]);self.assertEqual(state,before)
    def test_same_response_used_for_repeated_comm(self):
        planner=self.m.R27Tail({1,2});state=self.state(False)
        a=self.action();x=planner.apply(state,a,10.);y=planner.apply(x,a,10.)
        self.assertEqual(x[0][2]['w'],16.);self.assertEqual(y[0][2]['w'],21.);self.assertEqual(y[0][2]['comm_left'],1)
    def test_select_preserves_input_and_incumbent(self):
        state=self.state();before=copy.deepcopy(state);p=self.m.R27Tail({1,2},seconds=60,max_work=100000)
        _,rec=p.select(state,6,3,self.action());self.assertEqual(state,before)
        self.assertIn(('comm',2,None),[x['key'] for x in rec['rows']]);self.assertTrue(rec['complete'])
    def test_hidden_keys_rejected(self):
        for key in ('r','latent','response'):
            state=self.state();state[0][1][key]=1
            with self.assertRaises(ValueError):self.m.R27Tail({1,2}).select(state,6,3,self.action())
    def test_work_and_time_bounds(self):
        for p in (self.m.R27Tail({1,2},max_work=0),self.m.R27Tail({1,2},seconds=0)):
            with self.assertRaises(TimeoutError):p.select(self.state(),6,3,self.action())
    def test_unknown_vertices_ineligible(self):
        p=self.m.R27Tail({2});xs=p.candidates(self.state(),6)
        self.assertTrue(all(a['target_node_1']==2 and a['action']!='cut' for a in xs))
    def test_negative_structural_actions_remain_candidates(self):
        state=self.state();state[0][1]['w']=9.;state[0][2]['w']=10.
        xs=self.m.R27Tail({1,2}).candidates(state,5)
        self.assertTrue(any(a['action']=='shield' and a['gain']<0 for a in xs))
    def test_no_remaining_budget_or_slots(self):
        p=self.m.R27Tail({1,2});state=self.state()
        for b,slots in ((1,3),(6,0)):
            value,a=p.value(state,b,slots);self.assertIsNone(a);self.assertEqual(value,self.m.r27_score(*state[:2])[0])
    def test_no_upgrade_above_six_or_non50(self):
        m=self.m
        for n,b in ((50,7.),(4,6.),(100,6.)):
            model=m.ParticipantSquadModel.__new__(m.ParticipantSquadModel);model.node_count=n
            expected=[self.action()]
            with patch.object(m._R27Base,'_learned_candidates',return_value=expected),patch.object(m,'R27Tail') as planner:
                self.assertEqual(model._learned_candidates(b),expected);planner.assert_not_called()
    def test_incomplete_falls_back(self):
        m=self.m;model=m.ParticipantSquadModel.__new__(m.ParticipantSquadModel);model.node_count=50
        model.operation_limit=120;model.operation_count=80;model.local_nodes=self.state()[0]
        model.llm_counter=type('Counter',(),dict(safe_limit=119,call_count=40))()
        expected=[self.action()]
        with patch.object(m._R27Base,'_learned_candidates',return_value=expected),patch.object(model,'completed_state',return_value=self.state()),patch.dict(m.R27_SETTINGS,seconds=0.):
            self.assertEqual(model._learned_candidates(6.),expected);self.assertFalse(model.r27_records[-1]['complete'])

if __name__=='__main__':unittest.main()
