import contextlib
import copy
import io
import itertools
import sys
import unittest
from pathlib import Path
import networkx as nx

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
from experiment_round3 import fast_model
from experiment_round5 import FiniteEnv, PublicEnv
from experiment_round7 import initialize
from mechanism_round5 import seed, score as forward_score
from round6_common import people
from round7_exchange import D5, allocate, optimize, action
from round7_scan import LateScanStop, RestrictedRollout

class ExchangeTests(unittest.TestCase):
    def state(self):
        nodes={1:dict(w=-12.,persona='中立',comm_left=2),
               2:dict(w=3.,persona='中立',comm_left=3),
               3:dict(w=15.,persona='中立',comm_left=1)}
        return nodes,{(1,2),(2,3)},{1:7.,2:19.,3:4.},{}

    def test_allocation_matches_exhaustive_forward_dynamics(self):
        state=self.state()
        for structures in ((),(('cut',1,2),),(('shield',1,0),)):
            for budget in (5,8,12):
                for slots in (2,5):
                    result=allocate(state,structures,budget,slots)
                    if result is None:continue
                    removed={u for k,u,v in structures if k=='shield'}
                    cuts={(u,v) for k,u,v in structures if k=='cut'}
                    ids=[i for i in state[0] if i not in removed]
                    cost=sum(5 if k=='shield' else 3 for k,u,v in structures)
                    best=-float('inf')
                    for counts in itertools.product(*(range(state[0][i]['comm_left']+1) for i in ids)):
                        if 2*sum(counts)+cost>budget or sum(counts)+len(structures)>slots:continue
                        ns=copy.deepcopy(state[0])
                        for i in removed:del ns[i]
                        for i,count in zip(ids,counts):
                            ns[i]['w']+=sum(state[2][i]*.5**(3-ns[i]['comm_left']+k) for k in range(count))
                        value=forward_score({'nodes':[dict(d,id=i) for i,d in ns.items()],
                            'edges':[e for e in state[1]-cuts if not set(e)&removed]})
                        best=max(best,value)
                    self.assertAlmostEqual(result['score'],best,places=8)

    def test_exchange_plan_feasible_and_does_not_mutate_public_state(self):
        state=self.state();before=copy.deepcopy(state)
        engine=D5.FiniteRollout(*state,D5.SEARCH,20)
        baseline=engine.candidates(state,18)[0]
        result=optimize(state,18,20,baseline)
        self.assertEqual(state,before)
        self.assertGreaterEqual(result['candidate_score']+1e-8,result['reference_score'])
        plan=allocate(state,result['structures'],18,20)
        self.assertLessEqual(plan['cost'],18)
        self.assertLessEqual(len(plan['actions']),20)
        current=state
        for a in plan['actions']:current=engine.apply(current,a,state[2],False)
        self.assertAlmostEqual(D5.score(*current[:2])[0],plan['score'],places=8)

    def test_zero_search_budget_falls_back(self):
        result=optimize(self.state(),18,20,action('comm',2),
                        dict(seconds=0,evaluations=96,rounds=2,pool_per_kind=8,margin=.5))
        self.assertFalse(result['selected'])
        self.assertEqual(result['reason'],'reference_timeout')

class SmallStop(LateScanStop):scan_target=4

class ScanTests(unittest.TestCase):
    def make_model(self):
        graph=nx.path_graph(range(1,6))
        raw=FiniteEnv(seed(graph,{1:-100.,2:-100.,3:10.,4:10.,5:500.},r=1.))
        ps=copy.deepcopy(people());ps[0].setdefault('runtime',{})['node_count']=5
        model=initialize(fast_model(PublicEnv(raw),ps,SmallStop))
        for node in range(1,5):model._execute_scan(node)
        self.assertFalse(model._should_scan(raw.get_remaining_budget()))
        return raw,model

    def test_shield_keeps_intervention_phase_and_structure_search(self):
        raw,model=self.make_model()
        model._execute_planned_action(action('shield',1),raw.get_remaining_budget(),None)
        self.assertIsNone(model._next_scan_target())
        self.assertFalse(model._should_scan(raw.get_remaining_budget()))
        model._learned_candidates(raw.get_remaining_budget())
        record=model.scan_diagnostic_records[-1]
        self.assertEqual(record['known_alive'],3)
        self.assertEqual(record['unseen'],1)
        self.assertEqual(record['graph_vertices'],4)
        self.assertGreater(record['structural_candidates'],0)
        self.assertFalse(any(x.get('mode')=='plan' for x in model.decision_history))

    def test_unseen_vertex_preserved_without_hidden_attributes_or_targets(self):
        raw,model=self.make_model();state=model.completed_state()
        self.assertIn(5,state[0]);self.assertIn((4,5),state[1])
        self.assertEqual(state[0][5]['w'],-45.)
        self.assertNotEqual(state[0][5]['w'],raw._nodes[5]['w'])
        engine=RestrictedRollout(*state,D5.SEARCH,100,eligible=model.local_nodes)
        for a in engine.candidates(state,80,True):
            self.assertNotEqual(a['target_node_1'],5)
            self.assertNotEqual(a['target_node_2'],5)

if __name__=='__main__':unittest.main()
