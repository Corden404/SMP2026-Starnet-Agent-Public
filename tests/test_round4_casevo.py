import contextlib
import io
import unittest

from test_offline_baseline import FakeLLM, StrictFakeEnv, person_list
from starnet_model import ParticipantSquadModel


class Round4CaseVOTests(unittest.TestCase):
    def test_new_modes_execute_real_casevo_chains_on_all_sizes(self):
        for mode in ('rollout','information'):
            for n,budget,limit in ((4,60,50),(50,100,120),(100,200,250)):
                with self.subTest(mode=mode,n=n):
                    env=StrictFakeEnv(n,budget);people=person_list(n,limit,limit)
                    people[0]['runtime'].update(strategy='planner',planner={'decision_mode':mode,'rollout_seconds':0})
                    llm=FakeLLM(['{"index":0}']*250)
                    with contextlib.redirect_stdout(io.StringIO()):
                        model=ParticipantSquadModel(env,people,llm)
                        for step in range(limit):
                            if model.step():break
                        else:self.fail('Fuse reached')
                    self.assertLess(step+1,limit)
                    self.assertLess(model.llm_counter.call_count,limit)
                    self.assertEqual(sum(a[0]=='scan' for a in env._actions),n)
                    self.assertTrue(model.decision_history)
                    self.assertEqual(model.commander.chains['plan'].status,'finish')


if __name__=='__main__':unittest.main()
