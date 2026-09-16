import contextlib
import copy
import io
import itertools
import json
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_offline_baseline import FakeLLM, StrictFakeEnv, person_list
from starnet_model import ParticipantSquadModel
from experiment_round3 import ProxyEnv, fast_model, seed_for, VARIANTS
from zhipu import APICreditExhausted

class PlannerTests(unittest.TestCase):
    def fast(self, n=4, budget=10, options=None):
        env=StrictFakeEnv(n,budget)
        people=person_list(n,120,120)
        people[0]['runtime'].update({'strategy':'planner','planner':options or {}})
        return fast_model(env,people)

    def test_fast_estimator_matches_reference_on_disconnected_graph(self):
        rng=random.Random(901)
        for n in (1,4,20):
            nodes={i:{'w':rng.uniform(-20,30)} for i in range(n)}
            edges={(i,j) for i in nodes for j in nodes if i<j and rng.random()<.12}
            a,ia=ParticipantSquadModel.estimate_score(nodes,edges)
            b,ib=ParticipantSquadModel._fast_score(nodes,edges)
            self.assertAlmostEqual(a,b,places=9)
            for i in nodes:self.assertAlmostEqual(ia[i],ib[i])

    def test_fixed_graph_allocation_matches_exhaustive_integer_search(self):
        m=self.fast()
        nodes={1:{'w':-8.,'persona':'和平','comm_left':3},
               2:{'w':2.,'persona':'中立','comm_left':2},
               3:{'w':10.,'persona':'暴力','comm_left':1}}
        m.comm_effects={1:7.,2:12.,3:5.}
        edges={(1,2),(2,3)}
        for budget in (0,1,2,3,4,5,6,7,9,12):
            plan=m._terminal_plan(nodes,edges,budget)
            best=-float('inf')
            for counts in itertools.product(range(4),range(3),range(2)):
                if 2*sum(counts)>budget:continue
                candidate=copy.deepcopy(nodes)
                for i,k in enumerate(counts,1):
                    used=3-nodes[i]['comm_left']
                    candidate[i]['w']+=sum(m.comm_effects[i]*.5**(used+j) for j in range(k))
                best=max(best,m.estimate_score(candidate,edges)[0])
            self.assertAlmostEqual(plan['predicted_final'],best)
            self.assertGreaterEqual(plan['remaining_budget'],0)

    def test_planner_observes_action_and_llm_slots(self):
        m=self.fast()
        m.local_nodes={1:{'w':0.,'persona':'和平','comm_left':3}}
        m.operation_count=m.operation_limit-2
        p=m._terminal_plan(m.local_nodes,set(),100)
        self.assertEqual(sum(p['allocation'].values()),1)
        m.llm_counter.call_count=m.llm_counter.safe_limit
        p=m._terminal_plan(m.local_nodes,set(),100)
        self.assertEqual(sum(p['allocation'].values()),0)

    def test_persona_learning_is_shared_but_not_repeatedly_counted(self):
        m=self.fast()
        d={'w':0.,'persona':'和平','comm_left':3}
        initial=m._effect_estimate(2,d)[0]
        m.persona_observations={'和平':{1:4.}}
        self.assertLess(m._effect_estimate(2,d)[0],initial)
        m.persona_observations['和平'][1]=4.
        self.assertEqual(len(m.persona_observations['和平']),1)
        m.comm_effects[2]=30.
        self.assertEqual(m._effect_estimate(2,d),(30.,0.))
        m.planner_options['learning']=False
        self.assertEqual(m._effect_estimate(3,d)[0],initial)

    def test_incomplete_graph_never_plans_shield_or_cut(self):
        m=self.fast(n=4)
        m.local_nodes={1:{'w':-50.,'persona':'暴力','comm_left':3},
                       2:{'w':10.,'persona':'和平','comm_left':3}}
        m.local_edges={(1,2),(1,3)}
        plans=m._planning_plans(10)
        self.assertTrue(plans)
        self.assertTrue(all(p['structure']==[] and p['first']['action']=='comm' for p in plans))

    def test_structure_search_beats_only_communication_on_negative_hub(self):
        m=self.fast(n=4,budget=10)
        m.local_nodes={i:{'w':-100. if i==1 else 8.,'persona':'暴力' if i==1 else '和平','comm_left':3} for i in range(1,5)}
        m.local_edges={(1,2),(1,3),(1,4)}
        pure=m._terminal_plan(m.local_nodes,m.local_edges,10)
        plans=m._planning_plans(10)
        self.assertEqual(plans[0]['first']['action'],'shield')
        self.assertEqual(plans[0]['first']['target_node_1'],1)
        self.assertGreater(plans[0]['predicted_final'],pure['predicted_final'])

    def test_invalid_planner_options_rejected(self):
        for options in ({'learning':'yes'},{'depth':10},{'beam_width':0},{'scan_policy':'mystery'},{'plan_tolerance':float('nan')}):
            with self.subTest(options=options),self.assertRaises(ValueError):self.fast(options=options)

    def test_actual_casevo_planner_on_4_50_100_and_malformed_json(self):
        for n,budget,limit in ((4,60,50),(50,100,120),(100,200,250)):
            with self.subTest(n=n):
                env=StrictFakeEnv(n,budget)
                people=person_list(n,limit,limit)
                people[0]['runtime'].update({'strategy':'planner'})
                llm=FakeLLM(['{"index":999}','not-json','{"targets":[true,999]}'])
                with contextlib.redirect_stdout(io.StringIO()):
                    m=ParticipantSquadModel(env,people,llm)
                    for step in range(limit):
                        if m.step():break
                    else:self.fail('Fuse reached')
                self.assertLess(step+1,limit)
                self.assertLess(m.llm_counter.call_count,limit)
                self.assertEqual(len([a for a in env._actions if a[0]=='scan']),n)
                self.assertGreaterEqual(env.get_remaining_budget(),0)
                previous=len(env._actions)
                self.assertEqual(m.step(),1)
                self.assertEqual(len(env._actions),previous)

    def test_credit_exhaustion_escapes_new_planner_chain(self):
        class EmptyCreditLLM(FakeLLM):
            def send_message(self,*args,**kwargs):raise APICreditExhausted('test only')
        people=person_list()
        people[0]['runtime']['strategy']='planner'
        with contextlib.redirect_stdout(io.StringIO()):
            m=ParticipantSquadModel(StrictFakeEnv(),people,EmptyCreditLLM())
            with self.assertRaises(APICreditExhausted):m.step()
        self.assertEqual(m.llm_counter.call_count,1)

    def test_development_and_holdout_are_disjoint_and_reproducible(self):
        self.assertEqual(seed_for('dev-000'),seed_for('dev-000'))
        self.assertNotEqual(seed_for('dev-000'),seed_for('holdout-000'))
        self.assertEqual(len(seed_for('stress-000')['nodes']),100)

if __name__=='__main__':unittest.main()
