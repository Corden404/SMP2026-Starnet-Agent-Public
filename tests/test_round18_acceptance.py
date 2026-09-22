"""Live checkpoint sync, recovery refusal, stage gates and 4/100-node compatibility."""
import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round18_state as s
import round18_common as h
import round18_resume as r
import round18_closeout as close

class AcceptanceTests(unittest.TestCase):
    def test_progress_updates_both_json_and_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'out';plan=Path(directory)/'plan.md'
            plan.write_text('policy\n<!-- PROGRESS_START -->old<!-- PROGRESS_END -->\nend',encoding='utf8')
            with patch.object(s,'OUT',out),patch.object(s,'PLAN',plan):
                s.update('P1','COMPLETE','example completed',next_action='next')
                self.assertEqual(s.read(out/'progress.json')['steps']['P1']['status'],'COMPLETE')
                self.assertIn('example completed',plan.read_text(encoding='utf8'))
                self.assertEqual(len((out/'events.jsonl').read_text(encoding='utf8').splitlines()),1)
    def test_live_pid_detection_does_not_signal_process(self):self.assertTrue(r.pid_alive(os.getpid()))
    def test_recovery_refuses_nonlocal_paths(self):
        for path in ('../outside.json','real/runs/a.json','study/protocol.json','diagnostic/graphs/a.json'):
            with self.assertRaises(ValueError):r.local_target(path)
    def test_no_candidate_means_confirm_real_not_required(self):
        req=close.required_cohorts({'variants':['PC2','Cut1','Fast']},{'selected':None},None)
        self.assertEqual(req,{'dev':162,'integration':18})
    def test_real_is_required_only_after_local_gate(self):
        p={'variants':['PC2','Cut1','Fast']};dev={'selected':'Cut1'}
        self.assertNotIn('real',close.required_cohorts(p,dev,{'eligible_real':False}))
        self.assertEqual(close.required_cohorts(p,dev,{'eligible_real':True})['real'],96)
    def test_no_package_on_incomplete_or_nonpositive_real(self):
        a={'status':'COMPLETE','local':{'eligible_real':True}};real={'status':'COMPLETE','statistics':{'mean':.01}}
        self.assertTrue(close.delivery_allowed(a,real))
        self.assertFalse(close.delivery_allowed(dict(a,status='INCOMPLETE'),real))
        self.assertFalse(close.delivery_allowed(a,dict(real,statistics={'mean':0.})))
        self.assertFalse(close.delivery_allowed(dict(a,local=None),real))
    def test_cut_windows_once_and_other_sizes_bypass(self):
        m=h.load('Cut1','acceptance');first=dict(action='comm',target_node_1=1,target_node_2=None,prompt_id=1,cost=2.,gain=1.,roi=.5)
        cut=dict(action='cut',target_node_1=1,target_node_2=2,prompt_id=None,cost=3.,gain=1.,roi=1/3)
        state=({i:dict(w=float(i),persona='中立',comm_left=3) for i in (1,2)}, {(1,2)}, {},{})
        model=m.ParticipantSquadModel.__new__(m.ParticipantSquadModel)
        model.node_count=50;model.operation_count=60;model.operation_limit=120
        model.llm_counter=SimpleNamespace(safe_limit=119,call_count=30)
        model.local_nodes=state[0];model.completed_state=lambda:copy.deepcopy(state)
        with patch.object(m._R18CutBase,'_learned_candidates',return_value=[first]),\
             patch.object(m,'r18_best_cut',return_value=cut) as choose,\
             patch.object(m,'r18_cut_compare',return_value=(cut,dict(replaced=True,completed_worlds=3))):
            for budget in (19,18,9,8):model._learned_candidates(budget)
            self.assertEqual(choose.call_count,2)
            for n in (4,100):
                other=m.ParticipantSquadModel.__new__(m.ParticipantSquadModel);other.node_count=n
                self.assertEqual(other._learned_candidates(8),[first])
            self.assertEqual(choose.call_count,2)
    def test_actual_casevo_four_node_complete(self):
        spec=dict(h.specs('integration')[0],n=4)
        for variant in ('PC2','Cut1','Fast'):
            row=h.run_game(spec,variant,mode='integration')
            self.assertEqual(row['scan_count'],4);self.assertLess(row['logical_calls'],50)
            self.assertEqual(row['candidate_records'],[])
    def test_actual_casevo_hundred_node_boundary(self):
        spec=dict(h.specs('integration')[0],n=100)
        for variant in ('Cut1','Fast'):
            module=h.load(variant,'boundary');raw=h.FiniteEnv(h.c.generate(spec));logged=h.LoggedEnv(raw)
            with h.c.offline():
                model=h.initialize(module,logged,n=100,mode='integration')
                for _ in range(4):model.step()
            self.assertEqual(model.scan_target,100);self.assertEqual(model.pc2_records,[])
            self.assertEqual(len(logged.actions),4)

if __name__=='__main__':unittest.main()
