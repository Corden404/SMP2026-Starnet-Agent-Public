"""No paid requests; regressions for the exact additive scan variants."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
from round9_common import OUT, VARIANTS, load, people, source_files, specs, verify_originals
from round9_runner import initialize
from round4_seeds import generate
from experiment_round3 import fast_model, TopChoiceLLM
from experiment_round5 import FiniteEnv, PublicEnv


class Round9Tests(unittest.TestCase):
    def model(self,variant):
        raw=FiniteEnv(generate(specs('dev')[0]))
        module=load(variant)
        m=initialize(fast_model(PublicEnv(raw),copy.deepcopy(people()),module.ParticipantSquadModel))
        m.llm_counter=module.CountingLLM(TopChoiceLLM(),m.llm_limit)
        return raw,m

    def test_exact_baseline_and_additive_package(self):
        import zipfile
        from round9_common import ARCHIVE
        with zipfile.ZipFile(ARCHIVE) as z:
            self.assertEqual(source_files('S44'),{n:z.read(n) for n in z.namelist()})
        for v in VARIANTS:
            files=source_files(v)
            self.assertEqual(len(files),9)
            self.assertEqual(files['config.json'],source_files('S44')['config.json'])
            self.assertTrue(files['starnet_model.py'].startswith(source_files('S44')['starnet_model.py']))

    def test_q44_reorders_only_approved_nodes(self):
        raw,m=self.model('Q44')
        m.local_nodes={1:dict(w=0.,persona='中立',comm_left=3)}
        m.local_edges={(1,8)};m.scan_queue=[7,8,9]
        self.assertTrue(m._should_scan(raw.get_remaining_budget()))
        self.assertEqual(m.scan_queue,[8,7,9])
        self.assertEqual(raw.get_remaining_budget(),100)

    def test_all_stop_targets_keep_hidden_nodes_and_structure_after_shield(self):
        for v in VARIANTS:
            raw,m=self.model(v)
            count=int(v[1:]) if v in ('S40','S42') else 44
            for i in range(1,count+1):m._execute_scan(i)
            self.assertFalse(m._should_scan(raw.get_remaining_budget()))
            state=m.completed_state()
            self.assertEqual(len(state[0]),50)
            self.assertTrue(all('r' not in d for d in state[0].values()))
            self.assertEqual(state[0][50]['comm_left'],0)
            m._dispatch_env_action('shield',1,None,None,raw.get_remaining_budget())
            self.assertFalse(m._should_scan(raw.get_remaining_budget()))
            self.assertIsNone(m._next_scan_target())
            self.assertEqual(len(m.completed_state()[0]),49)

    def test_b44_refreshes_at32_and_uses_two_candidate_batches(self):
        raw,m=self.model('B44');calls=[]
        original=m.scout._run_chain
        def record(name,data):
            calls.append((m._known_count(),copy.deepcopy(data)));return original(name,data)
        m.scout._run_chain=record
        for _ in range(44):self.assertEqual(m.step(),0)
        late=[(n,d) for n,d in calls if n>=32]
        self.assertEqual([n for n,d in late],[32,34,36,38,40,42])
        self.assertTrue(all(len(d['unknown_ids'])==2 for n,d in late))
        self.assertEqual(len(raw._scanned),44)

    def test_information_is_public_and_remainder_explicit(self):
        raw,m=self.model('I44');calls=[]
        original=m.scout._run_chain
        def record(name,data):calls.append(copy.deepcopy(data));return original(name,data)
        m.scout._run_chain=record
        for _ in range(44):m.step()
        self.assertEqual(calls[-1]['remaining_scans'],4)
        for d in calls:
            features=json.loads(d['candidate_features'])
            self.assertEqual([x['id'] for x in features],d['unknown_ids'])
            self.assertTrue(all(set(x)=={'id','rank','observed_degree','scanned_neighbors'} for x in features))

    def test_invalid_scout_targets_fall_back_without_illegal_actions(self):
        for v in ('B44','I44'):
            raw,m=self.model(v)
            m.scout._run_chain=lambda *a:{'targets':[True,999,1,1,'2',-1]}
            m.step()
            self.assertEqual(raw._scanned,{1})

    def test_real_casevo_and_fast_scanning_match(self):
        for v in VARIANTS:
            raw,fast=self.model(v)
            other=FiniteEnv(generate(specs('dev')[0]))
            with contextlib.redirect_stdout(io.StringIO()):
                full=load(v).ParticipantSquadModel(PublicEnv(other),copy.deepcopy(people()),TopChoiceLLM())
                for _ in range(fast.scan_target):
                    self.assertEqual(fast.step(),full.step())
                    self.assertEqual(fast.local_nodes,full.local_nodes)
                    self.assertEqual(fast.scan_queue,full.scan_queue)
                self.assertEqual(fast.llm_counter.call_count,full.llm_counter.call_count)

    def test_other_sizes_keep_full_scan(self):
        for n in (4,100):
            for v in VARIANTS:
                s=dict(specs('dev')[2],n=n)
                ps=copy.deepcopy(people());ps[0]['runtime']['node_count']=n
                with contextlib.redirect_stdout(io.StringIO()):
                    m=load(v).ParticipantSquadModel(PublicEnv(FiniteEnv(generate(s))),ps,TopChoiceLLM())
                self.assertEqual(m.scan_target,n)

    def test_original_files_unchanged(self):
        self.assertEqual(verify_originals()['status'],'PASS')

if __name__=='__main__':unittest.main()
