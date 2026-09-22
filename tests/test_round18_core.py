"""Round18 archive, durable state, public replay and frozen design tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round18_state as s
import round18_common as h

class CoreTests(unittest.TestCase):
    def test_baseline_exact_nine_files(self):
        files=s.archive_files();self.assertEqual(len(files),9)
        self.assertEqual(files,h.source_files('PC2'))
    def test_design_counts_and_crossing(self):
        for pool,count in [('diagnostic',12),('dev',54),('confirm',216),('stress',54),('real',24)]:
            rows=h.specs(pool);self.assertEqual(len(rows),count)
            self.assertEqual(len({r['case'] for r in rows}),count)
        self.assertEqual(len({(r['family'],r['scale'],r['persona_mix']) for r in h.specs('dev')}),54)
    def test_seed_identities_new_and_disjoint(self):
        entries=h.entries('diagnostic')
        self.assertEqual(len({e['sha256'] for e in entries}),12)
        self.assertTrue(all(e['spec']['case'].startswith('r18-') for e in entries))
    def test_atomic_json_and_immutable_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'x.json';s.write(p,{'n':1},True)
            with self.assertRaises(RuntimeError):s.write(p,{'n':2},True)
            self.assertEqual(s.read(p),{'n':1});s.write(p,{'n':3})
            self.assertEqual(s.read(p),{'n':3});self.assertEqual(len(list(Path(directory).glob('*.tmp'))),0)
    def test_completed_record_reuses_without_running(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'r.json';calls=[]
            op=lambda:(calls.append(1) or {'value':9})
            h.checked_record(p,{'seed':1},op);h.checked_record(p,{'seed':1},op)
            self.assertEqual(calls,[1])
            with self.assertRaises(RuntimeError):h.checked_record(p,{'seed':2},op)
    def test_failed_record_is_preserved_no_implicit_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'r.json'
            def fail():raise ValueError('controlled')
            with self.assertRaises(ValueError):h.checked_record(p,{'seed':1},fail)
            self.assertEqual(s.read(p)['status'],'FAILED')
            with self.assertRaises(RuntimeError):h.checked_record(p,{'seed':1},lambda:{})
    def test_started_record_blocks_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'r.json';s.write(p,dict(status='STARTED',identity={'seed':1}))
            with self.assertRaises(RuntimeError):h.checked_record(p,{'seed':1},lambda:{})
    def test_independent_module_caches(self):
        a=h.load('PC2','test-a');b=h.load('PC2','test-b')
        self.assertIsNot(a.coefficients,b.coefficients)
        self.assertIsNot(a.pc2_coefficients,b.pc2_coefficients)
    def test_restored_consumed_windows_and_public_wrapper(self):
        m=h.load('PC2','test-state')
        row=dict(spec=dict(h.specs('integration')[0],n=4),actions=[],chains=[],candidate_records=[])
        snapshot=dict(operation=0,calls=0,budget=8.,pc2_seen=['first','budget20'])
        # restore itself is tested with the actual four-node public environment.
        with h.c.offline():
            module,raw,logged,model=h.restore(row,snapshot,'test-restore')
        self.assertEqual(model._pc2_seen,{'first','budget20'})
        self.assertFalse(hasattr(model.env,'_nodes'))
        self.assertFalse(hasattr(model.env,'_edges'))
    def test_no_hidden_keys_in_public_snapshot(self):
        seed=h.c.generate(h.specs('diagnostic')[0]);raw=h.FiniteEnv(seed)
        public=h.PublicEnv(raw).scan_node(1)
        self.assertEqual(set(public),{'w','persona','comm_left','neighbors'})
    def test_branch_helper_reserves_forced_call(self):
        from types import SimpleNamespace
        seen=[]
        model=SimpleNamespace(llm_counter=SimpleNamespace(_reserve_call=lambda:seen.append('call')),
            local_nodes={1:dict(w=1)},action_history=[{'success':True}],
            _execute_planned_action=lambda a,b,before:seen.append('action'))
        h.c.force(model,SimpleNamespace(get_remaining_budget=lambda:10),dict(action='comm',target_node_1=1))
        self.assertEqual(seen,['call','action'])

if __name__=='__main__':unittest.main()
