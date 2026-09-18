import copy
import math
import random
import time
import unittest
from pathlib import Path

from test_offline_baseline import PROJECT_ROOT, person_list, StrictFakeEnv
from experiment_round3 import fast_model

from starnet_model import BudgetRollout


class RolloutTests(unittest.TestCase):
    def engine(self,n=4,**options):
        nodes={i:{'w':-40. if i==1 else float(i),'persona':'暴力' if i==1 else '和平','comm_left':3} for i in range(1,n+1)}
        edges={(1,i) for i in range(2,n+1)}
        opts={'rollout_samples':1,'rollout_width':12,'rollout_seconds':20,**options}
        return BudgetRollout(nodes,edges,{}, {},opts,30)

    def test_fast_all_action_gains_match_existing_estimator(self):
        rng=random.Random(817)
        for n in (3,8,20):
            eng=self.engine(n)
            eng.initial[0][1]['w']=rng.uniform(-50,20)
            eng.initial[1].update((u,v) for u in range(1,n+1) for v in range(u+1,n+1) if rng.random()<.15)
            people=person_list(n,120,120);people[0]['runtime'].update(strategy='planner',planner={'decision_mode':'roi'})
            m=fast_model(StrictFakeEnv(n,100),people)
            m.local_nodes=copy.deepcopy(eng.initial[0]);m.local_edges=set(eng.initial[1])
            expected={eng.key(a):a['gain'] for a in m._learned_candidates(75)}
            actual={eng.key(a):a['gain'] for a in eng.candidates(eng.initial,75)}
            self.assertEqual(set(expected),set(actual))
            for key in expected:self.assertAlmostEqual(expected[key],actual[key],places=9)

    def test_state_isolation_fixed_replay_and_zero_search(self):
        eng=self.engine();before=copy.deepcopy(eng.initial)
        a=eng.choose(10);b=self.engine().choose(10)
        self.assertEqual(eng.initial,before)
        self.assertEqual(a[0]['first'],b[0]['first'])
        self.assertEqual(a[0]['predicted_final'],b[0]['predicted_final'])
        eng.options['rollout_seconds']=0
        self.assertEqual(eng.choose(10)[0]['first'],eng.candidates(eng.initial,10)[0])

    def test_budget_and_slots_and_learning_visibility(self):
        eng=self.engine();latent={i:7. for i in eng.initial[0]}
        comm=next(a for a in eng.candidates(eng.initial,10) if a['action']=='comm')
        nolearn=eng.apply(eng.initial,comm,latent,False)
        learned=eng.apply(eng.initial,comm,latent,True)
        self.assertFalse(nolearn[2]);self.assertEqual(learned[2],{comm['target_node_1']:7.})
        for budget in (2,3,5,7):
            p=eng.continuation(eng.initial,budget,1,latent,True)
            self.assertLessEqual(len(p['actions']),1)
            self.assertGreaterEqual(p['remaining_budget'],0)
        eng.slots=0;self.assertEqual(eng.choose(20),[])

    def test_two_step_small_graph_matches_exhaustive_with_full_width(self):
        eng=self.engine(n=3,rollout_combinations=True,rollout_margin=0.)
        latent={i:eng.effect(i,d,{}, {})[0] for i,d in eng.initial[0].items()}
        eng.slots=2
        for budget in (2,3,5,7):
            best=eng.metrics(eng.initial[0],eng.initial[1])[0]
            for a in eng.candidates(eng.initial,budget,negative=True):
                state=eng.apply(eng.initial,a,latent,False)
                best=max(best,eng.metrics(state[0],state[1])[0])
                for b in eng.candidates(state,budget-a['cost'],negative=True):
                    next_state=eng.apply(state,b,latent,False)
                    best=max(best,eng.metrics(next_state[0],next_state[1])[0])
            plans=eng.choose(budget)
            self.assertAlmostEqual(plans[0]['predicted_final'],best)


if __name__=='__main__':unittest.main()
