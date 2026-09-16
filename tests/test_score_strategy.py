import contextlib
import io
import json
import unittest
from unittest.mock import patch

from test_offline_baseline import FakeLLM, StrictFakeEnv, make_model, person_list
from starnet_model import ParticipantSquadModel
from zhipu import APICreditExhausted, ZhipuLLM

class ScoreStrategyTests(unittest.TestCase):
    def score_model(self, env, llm=None, n=4, limit=120):
        people=person_list(node_count=n,operation_limit=limit,llm_limit=limit)
        people[0]["runtime"]["strategy"]="score"
        return ParticipantSquadModel(env,people,llm or FakeLLM())

    def test_credit_exhaustion_escapes_chain_without_retry(self):
        class Response:
            status_code=429
            text='{"error":{"code":"1113","message":"余额不足"}}'
            def json(self):
                return json.loads(self.text)
        llm=ZhipuLLM("offline-placeholder",5)
        model,env,_=make_model(llm=llm)
        with patch("zhipu.requests.post",return_value=Response()) as post:
            with self.assertRaises(APICreditExhausted):
                model.step()
        self.assertEqual(post.call_count,1)
        self.assertEqual(env._actions,[])
        self.assertEqual(model.llm_counter.call_count,1)

    def test_no_harmful_endgame_cut(self):
        model=self.score_model(StrictFakeEnv())
        model.local_nodes={1:{"w":49.375,"persona":"和平","comm_left":0},
                           3:{"w":26.25,"persona":"中立","comm_left":0},
                           4:{"w":46.5,"persona":"和平","comm_left":0}}
        model.dead_nodes={2}
        model.local_edges={(1,4),(3,4)}
        self.assertEqual(model.score_candidates(40),[])
        self.assertEqual(model.step(),1)
        self.assertEqual(model.env._actions,[])

    def test_dangerous_cut_is_rejected_by_candidate_generation(self):
        model=self.score_model(StrictFakeEnv(node_count=3),n=3)
        model.local_nodes={i:{"w":w,"persona":"和平","comm_left":0}
                           for i,w in ((1,0.0),(2,100.0),(3,0.0))}
        model.local_edges={(1,2),(2,3)}
        self.assertEqual(model.score_candidates(3),[])

    def test_batch_scan_and_score_mode_complete_within_limits(self):
        for n,budget,limit in ((4,60,50),(50,100,120),(100,200,250)):
            with self.subTest(n=n):
                env=StrictFakeEnv(node_count=n,budget=budget)
                class LLM(FakeLLM):
                    def send_message(self,prompt,json_flag=False):
                        self.prompts.append(prompt)
                        if '"targets"' in prompt:
                            return json.dumps({"targets":list(range(1,n+1))})
                        return '{"index":0}'
                model=self.score_model(env,LLM(),n,limit)
                calls=0
                with contextlib.redirect_stdout(io.StringIO()):
                    while calls<=limit:
                        calls+=1
                        if model.step():break
                self.assertLess(calls,limit)
                self.assertLess(model.llm_counter.call_count,limit)
                self.assertGreaterEqual(env.get_remaining_budget(),0)
                self.assertEqual(len([a for a in env._actions if a[0]=="scan"]),n)
                self.assertTrue(all(1<=a[1]<=n for a in env._actions))
                if n>=50:self.assertLess(env.get_remaining_budget(),2)

if __name__=="__main__":unittest.main()
