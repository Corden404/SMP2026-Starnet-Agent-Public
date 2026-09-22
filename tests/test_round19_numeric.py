"""Round19 kernel contracts, information boundaries, window state and frozen-PC2 equivalence."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round19_common as h
import round19_state as s


class NumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=h.load('Numeric','numeric_tests');cls.base=h.load('PC2','numeric_reference')
    def setUp(self):h.clear(self.m);h.clear(self.base)
    def state(self):
        return ({1:dict(w=4.,persona='和平',comm_left=3),2:dict(w=-8.,persona='暴力',comm_left=3),
                 3:dict(w=1.,persona='中立',comm_left=2),4:dict(w=3.,persona='unobserved',comm_left=0)},
                {(1,2),(2,3),(3,4)}, {}, {})
    def ctx(self):return dict(operation=60,calls=25,operation_limit=120,safe_limit=119,seen={'first','budget40','budget20'})
    def test_base_archive_exact(self):
        self.assertEqual(h.source_files('PC2'),s.archive_files())
    def test_private_candidate_values_equal(self):
        rng=random.Random(1901)
        for n in (4,13,50,100):
            nodes={i:dict(w=rng.uniform(-50,30),persona=['和平','中立','暴力'][i%3],comm_left=3) for i in range(1,n+1)}
            edges={(i,j) for i in nodes for j in nodes if i<j and rng.random()<.07};state=(nodes,edges,{},{});eligible=set(nodes)-{n}
            ref=self.m.R18CutRollout(*state,self.m.SEARCH,12,eligible=eligible)
            fast=self.m.R19Kernel(*state,self.m.SEARCH,12,eligible=eligible)
            for budget in (2,3,5,20):self.assertEqual(ref.candidates(state,budget),fast.candidates(state,budget))
    def test_zero_deadline_keeps_incumbent(self):
        a=h.action(('comm',1,None));b=h.action(('cut',1,2))
        got,e=self.m.r19_compare(self.state(),9,{1,2,3},self.ctx(),a,b,seconds=0)
        self.assertEqual(a,got);self.assertFalse(e['replaced'])
    def test_partial_mean_cannot_change(self):
        a=h.action(('comm',1,None));b=h.action(('cut',1,2))
        with patch.object(self.m,'r19_continue',side_effect=[{'score':0.},{'score':99.},TimeoutError()]):
            got,e=self.m.r19_compare(self.state(),9,{1,2,3},self.ctx(),a,b)
        self.assertEqual(got,a);self.assertEqual(e['completed_worlds'],1);self.assertFalse(e['replaced'])
    def test_same_action_same_world_zero(self):
        a=h.action(('comm',1,None))
        got,e=self.m.r19_compare(self.state(),4,{1,2,3},self.ctx(),a,a)
        self.assertEqual(e['predicted_delta'],0.);self.assertFalse(e['replaced'])
    def test_risk_is_separate_rule(self):
        a=h.action(('comm',1,None));b=h.action(('cut',1,2))
        for risk,expected in ((False,b),(True,a)):
            # Ordering is incumbent/challenger, challenger/incumbent, incumbent/challenger.
            with patch.object(self.m,'r19_continue',side_effect=[{'score':0.},{'score':3.},{'score':-1.},{'score':0.},{'score':0.},{'score':3.}]):
                got,e=self.m.r19_compare(self.state(),9,{1,2,3},self.ctx(),a,b,risk=risk)
            self.assertEqual(got,expected)
    def test_state_and_context_unchanged(self):
        state=self.state();ctx=self.ctx();original=copy.deepcopy((state,ctx))
        self.m.r19_compare(state,4,{1,2,3},ctx,h.action(('comm',1,None)),h.action(('comm',2,None)))
        self.assertEqual((state,ctx),original)
    def test_hidden_input_is_rejected(self):
        state=self.state();state[0][1]['r']=4.
        with self.assertRaises(ValueError):self.m.r19_compare(state,9,{1,2,3},self.ctx(),h.action(('comm',1,None)),h.action(('cut',1,2)))
    def test_outer_world_only_delivers_selected_feedback(self):
        state=self.state();seen=[]
        def select(belief,budget,eligible,context,deadline,mode):seen.append(copy.deepcopy(belief));return None,None
        with patch.object(self.m,'r19_next',side_effect=select):
            self.m.r19_continue(state,4,{1,2,3},self.ctx(),h.action(('comm',1,None)),{1:7.,2:999.,3:1.,4:1.},time.monotonic()+8)
        self.assertEqual(seen[0][2],{1:7.});self.assertNotIn(2,seen[0][2]);self.assertEqual(seen[0][0][2]['w'],-8.)
    def test_consumed_window_not_repeated(self):
        ctx=self.ctx();ctx['seen'].remove('budget20')
        a,e=self.m.r19_next(self.state(),19,{1,2,3},ctx,time.monotonic()+8)
        self.assertEqual(e['tags'],['budget20'])
        _,e2=self.m.r19_next(self.state(),18,{1,2,3},ctx,time.monotonic()+8)
        self.assertIsNone(e2)
        _,e3=self.m.r19_next(self.state(),9,{1,2,3},ctx,time.monotonic()+8)
        self.assertEqual(e3['tags'],['budget10'])
    def test_budget_calls_operations_and_slots(self):
        ctx=self.ctx();a=h.action(('comm',1,None))
        out=self.m.r19_continue(self.state(),2,{1,2,3},ctx,a,{1:3.,2:1.,3:1.,4:1.},time.monotonic()+8)
        self.assertEqual(out['budget'],0.);self.assertEqual(out['calls'],ctx['calls']+1)
        self.assertEqual(out['operation'],ctx['operation']+1);self.assertEqual(out['slots'],self.m.r19_slots(ctx)-1)
    def test_illegal_targets_or_slots(self):
        for a,budget,slots in ((h.action(('cut',3,4)),5,2),(h.action(('comm',1,None)),1,2),(h.action(('comm',1,None)),2,0)):
            with self.assertRaises(ValueError):self.m.r19_legal(self.state(),a,budget,slots,{1,2,3})
    def test_private_cache_does_not_warm_base(self):
        before=(self.m.coefficients.cache_info(),self.m.pc2_coefficients.cache_info())
        self.m.r19_compare(self.state(),6,{1,2,3},self.ctx(),h.action(('comm',1,None)),h.action(('cut',1,2)))
        self.assertEqual(before,(self.m.coefficients.cache_info(),self.m.pc2_coefficients.cache_info()))
    def test_numpy_missing_isolated_fallback(self):
        with patch.object(self.m,'_pc_np',None):
            result=self.m.r18_cut_score(*self.state()[:2])[0]
            self.assertAlmostEqual(result,self.base.score(*self.state()[:2])[0],places=10)
        self.assertEqual(self.m.coefficients.cache_info().currsize,0)
    def test_topology_and_weight_cache_invalidation(self):
        state=self.state();eng=self.m.R19Kernel(*state,self.m.SEARCH,5,eligible={1,2,3})
        for action in (h.action(('comm',1,None)),h.action(('cut',1,2)),h.action(('shield',2,None))):
            ns=eng.apply(state,action,{1:7.,2:1.,3:1.,4:1.},True)
            ref=self.m.R18CutRollout(*ns,self.m.SEARCH,5,eligible={1,2,3})
            self.assertEqual(eng.candidates(ns,8),ref.candidates(ns,8))
    def test_depth2_forecast_matches_frozen(self):
        state=self.state();a=h.action(('comm',1,None));b=h.action(('cut',1,2))
        _,old=self.m.r18_cut_compare(state,9,self.m.r19_slots(self.ctx()),{1,2,3},a,b,seconds=8)
        _,new=self.m.r19_compare(state,9,{1,2,3},self.ctx(),a,b,seconds=8,mode='depth2')
        self.assertAlmostEqual(old['predicted_delta'],new['predicted_delta'],places=10)
    def test_full_pc2_select_matches_archived_six_historical_states(self):
        root=s.ROOT/'agent_notes/experiments/round18/diagnostic/baseline'
        for path in sorted(root.glob('*.json'))[:6]:
            row=s.read(path);snap=next(x for x in row['states'] if 'budget10' in x['tags'])
            module=self.base;h.clear(module)
            with h.old.c.offline():
                _,raw,logged,model=h.old.c.restore(row,snap['operation'],module)
                model._pc2_init();model._pc2_seen={t for e in row['candidate_records'] if e['operation']<snap['operation'] for t in e['tags']}
                state=model.completed_state();ctx=h.context(model)
                reference=model._learned_candidates(snap['budget'])
                got,event=self.m.r19_next(state,snap['budget'],set(model.local_nodes),ctx,time.monotonic()+8)
            self.assertEqual(module.BudgetRollout.key(reference[0]),module.BudgetRollout.key(got))
            self.assertEqual(ctx['seen'],model._pc2_seen)
    def test_completed_records_reused_failed_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';calls=[]
            h.checked(p,{'x':1},lambda:(calls.append(1) or {'value':3}))
            h.checked(p,{'x':1},lambda:calls.append(2));self.assertEqual(calls,[1])
            with self.assertRaises(RuntimeError):h.checked(p,{'x':2},lambda:None)
    def test_plan_progress_and_json_sync(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';plan=Path(d)/'plan.md';plan.write_text('<!-- EXECUTION_PROGRESS_START --><!-- EXECUTION_PROGRESS_END -->',encoding='utf8')
            with patch.object(s,'OUT',out),patch.object(s,'PLAN',plan):
                s.update('R1','COMPLETE','done')
                self.assertEqual(s.read(out/'progress.json')['steps']['R1']['status'],'COMPLETE')
                self.assertIn('done',plan.read_text(encoding='utf8'))

if __name__=='__main__':unittest.main()
