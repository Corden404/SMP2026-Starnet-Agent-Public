import contextlib
import copy
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
from round8_common import prepare_baseline, people, load_s48
prepare_baseline()
from round8_scan import Q48, S46, S44
from experiment_round8 import initialize, specs
from experiment_round3 import fast_model, TopChoiceLLM
from experiment_round5 import FiniteEnv, PublicEnv
from round4_seeds import generate

class Round8Tests(unittest.TestCase):
    def model(self, cls):
        raw=FiniteEnv(generate(specs('dev')[0]))
        return raw,initialize(fast_model(PublicEnv(raw),copy.deepcopy(people()),cls))

    def test_queue_refresh_uses_new_public_edges_without_extra_scan(self):
        raw,m=self.model(Q48)
        m.local_nodes={1:dict(w=10.,persona='和平',comm_left=3)}
        m.local_edges={(1,8)};m.scan_queue=[7,8,9]
        before=raw.get_remaining_budget()
        self.assertTrue(m._should_scan(before))
        self.assertEqual(m.scan_queue,[8,7,9])
        self.assertEqual(raw.get_remaining_budget(),before)
        m.local_nodes[2]=dict(w=0.,persona='中立',comm_left=3)
        m.local_nodes[3]=dict(w=0.,persona='中立',comm_left=3)
        m.local_edges.update({(2,9),(3,9)})
        self.assertTrue(m._should_scan(before))
        self.assertEqual(m.scan_queue,[9,8,7])

    def test_queue_refresh_drops_known_and_dead_targets(self):
        raw,m=self.model(Q48);m.local_nodes={1:dict(w=0.,persona='中立',comm_left=3)}
        m.dead_nodes={2};m.scan_queue=[1,2,4,3]
        m._should_scan(raw.get_remaining_budget())
        self.assertEqual(m.scan_queue,[3,4])

    def test_scan_coverage_and_hidden_information_boundary(self):
        for cls,coverage in ((Q48,48),(S46,46),(S44,44)):
            with self.subTest(cls=cls.__name__):
                raw,m=self.model(cls)
                for i in range(1,coverage+1):m._execute_scan(i)
                self.assertFalse(m._should_scan(raw.get_remaining_budget()))
                state=m.completed_state()
                self.assertEqual(len(state[0]),50)
                unknown=set(range(coverage+1,51))
                self.assertTrue(all(state[0][i]['comm_left']==0 for i in unknown))
                self.assertTrue(all('r' not in d for d in state[0].values()))
                m._dispatch_env_action('shield',1,None,None,raw.get_remaining_budget())
                self.assertFalse(m._should_scan(raw.get_remaining_budget()))
                self.assertIsNone(m._next_scan_target())
                self.assertEqual(len(m.completed_state()[0]),49)

    def test_true_casevo_and_fast_scan_sequences_match(self):
        data=generate(specs('glm-base')[2])
        a=FiniteEnv(data);b=FiniteEnv(data)
        with contextlib.redirect_stdout(io.StringIO()):
            fast=initialize(fast_model(PublicEnv(a),copy.deepcopy(people()),Q48))
            full=Q48(PublicEnv(b),copy.deepcopy(people()),TopChoiceLLM())
            for _ in range(48):
                self.assertEqual(fast.step(),full.step())
                self.assertEqual(fast.local_nodes,full.local_nodes)
                self.assertEqual(fast.scan_queue,full.scan_queue)
            self.assertEqual(fast.llm_counter.call_count,full.llm_counter.call_count)
        self.assertEqual(a._scanned,b._scanned)

    def test_size_boundary_reverts_to_complete_scan(self):
        for n in (4,100):
            ps=copy.deepcopy(people());ps[0]['runtime']['node_count']=n
            s=generate(specs('dev')[0]);s['global_setting']['max_budget']=2*n
            with contextlib.redirect_stdout(io.StringIO()):
                for cls in (Q48,S46,S44):
                    m=cls(PublicEnv(FiniteEnv(s)),ps,TopChoiceLLM())
                    self.assertEqual(m.scan_target,n)

if __name__=='__main__':unittest.main()
