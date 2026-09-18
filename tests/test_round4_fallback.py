import unittest
from unittest.mock import patch

from test_offline_baseline import StrictFakeEnv, person_list
from experiment_round3 import fast_model


class RolloutFallbackTests(unittest.TestCase):
    def test_invalid_llm_choice_returns_preserved_baseline(self):
        people=person_list(4,120,120)
        people[0]['runtime'].update(strategy='planner',planner={'decision_mode':'rollout'})
        model=fast_model(StrictFakeEnv(4,100),people)
        model.local_nodes={i:{'w':1.,'persona':'和平','comm_left':3} for i in range(1,5)}
        baseline=model._action_dict('comm',1)
        candidate=model._action_dict('comm',2)
        plans=[{'first':candidate,'predicted_final':20.,'baseline':False},
               {'first':baseline,'predicted_final':10.,'baseline':True}]
        with patch.object(model,'_should_scan',return_value=False),\
             patch.object(model,'_next_scan_target',return_value=None),\
             patch.object(model,'_rollout_plans',return_value=plans),\
             patch.object(model.commander,'_run_chain',return_value={'index':999}),\
             patch.object(model,'_execute_planned_action',return_value=1) as execute:
            model._planned_step()
            self.assertEqual(execute.call_args.args[0],baseline)
            self.assertTrue(model.decision_history[-1]['invalid_choice_baseline_fallback'])


if __name__=='__main__':unittest.main()
