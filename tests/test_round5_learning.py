import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workspace'))
import networkx as nx
from diagnostics_round5 import L0, LearningRollout, update_response
from experiment_round5 import OUT, FiniteEnv, PublicEnv
from experiment_round3 import fast_model
from mechanism_round5 import seed


class LearningConsistencyTests(unittest.TestCase):
    def test_real_and_simulated_feedback_use_same_update(self):
        s=seed(nx.path_graph(range(1,5)),{i:10. for i in range(1,5)},r=.2)
        env=FiniteEnv(s)
        people=json.loads((OUT/'baseline/config.json').read_text(encoding='utf-8'))['person']
        m=fast_model(PublicEnv(env),people,L0)
        for i in range(1,5):m._execute_scan(i)
        rollout=LearningRollout(m.local_nodes,m.local_edges,m.comm_effects,m.persona_observations,{},10)
        latent={i:3. for i in m.local_nodes};original=dict(latent)
        action=m._action_dict('comm',1);before=copy.deepcopy(m.local_nodes[1])
        simulated=rollout.apply(rollout.initial,action,latent,True)
        m._execute_planned_action(action,env.get_remaining_budget(),before)
        self.assertEqual(m.local_nodes,simulated[0])
        self.assertEqual(m.comm_effects,simulated[2])
        self.assertEqual(m.persona_observations,simulated[3])
        self.assertEqual(latent,original)

    def test_learning_toggle_keeps_first_action_latent_and_physical_state(self):
        nodes={i:{'w':0.,'persona':'中立','comm_left':3} for i in range(1,5)}
        roll=LearningRollout(nodes,{(1,2),(2,3)}, {},{}, {},10)
        action={'action':'comm','target_node_1':1,'target_node_2':None,'prompt_id':1,'cost':2.}
        latent={1:30.,2:3.,3:5.,4:10.}
        on=roll.apply(roll.initial,action,latent,True)
        off=roll.apply(roll.initial,action,latent,False)
        self.assertEqual(on[:2],off[:2]);self.assertNotEqual(on[2:],off[2:])
        again=roll.apply(on,action,latent,True)
        self.assertEqual(again[0][1]['w'],45.)
        self.assertEqual(again[2][1],30.)
        self.assertNotEqual(roll.effect(2,on[0][2],on[2],on[3]),roll.effect(2,off[0][2],off[2],off[3]))


if __name__=='__main__':unittest.main()
