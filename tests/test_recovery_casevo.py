import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from test_offline_baseline import person_list
from test_recovery_journal import Clock,response
from recovery_store import Store,Journal,InfrastructureFailure
from recovery_env import Gate,Transport,DurableEnv
from recovery_llm import DurableLLM,attach_observers
from recovery_runner import Health
from experiment_round3 import ProxyEnv
from team_submission.starnet_model import ParticipantSquadModel
import requests


class BusinessServer:
    def __init__(self,fail=True):self.envs={};self.fail=fail;self.communications=0
    def post(self,url,json,timeout):
        name=url.split('/api/')[-1]
        if name=='start_session':
            sid=str(len(self.envs)+1);self.envs[sid]=ProxyEnv(json['seed']);return response({'session_id':sid})
        e=self.envs[json['session_id']]
        if name=='get_budget':return response({'budget':e.get_remaining_budget()})
        if name=='scan':return response({'data':e.scan_node(json['node_id'])})
        if name=='communicate':
            result=e.communicate(json['node_id'],json['prompt_id']);self.communications+=1
            if self.fail:self.fail=False;raise requests.ReadTimeout('applied, response lost')
            return response(result)
        if name=='cut':return response({'success':e.cut_link(json['u'],json['v'])})
        if name=='shield':return response({'success':e.shield_node(json['node_id'])})
        if name=='evaluate':return response({'final_score':e.evaluate()})
        raise AssertionError(name)


class CaseVORecoveryTests(unittest.TestCase):
    def test_actual_chains_restore_same_actions_and_logical_counts(self):
        for mode in ('roi','rollout','information'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                store=Store(Path(temp)/'j.sqlite');clock=Clock();gate=Gate(store,clock.time,clock.sleep)
                store.put('circuit',{'phase':'READY'});tid=store.trial({'mode':mode});server=BusinessServer();paid=[]
                def post(url,headers,json,timeout):
                    paid.append(json)
                    return response({'choices':[{'message':{'content':'{"index":0,"targets":[]}'},'finish_reason':'stop'}],
                                     'id':str(len(paid)),'model':'glm-4-plus','usage':{'total_tokens':1}})
                final=None
                for attempt in range(2):
                    aid=store.attempt(tid);j=Journal(store,tid,aid)
                    env=DurableEnv('https://example.invalid',Health.seed(),j,Transport(store,gate,aid,post=server.post))
                    llm=DurableLLM('fake',j,gate,post=post);people=person_list(4,50,50)
                    people[0]['runtime'].update(strategy='planner',planner={'decision_mode':mode})
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            model=ParticipantSquadModel(env,people,llm);attach_observers(model,llm,j,env)
                            for step in range(50):
                                if model.step():break
                            else:self.fail('fuse')
                        final=env.trigger_eval();j.assert_end()
                    except InfrastructureFailure:
                        self.assertEqual(attempt,0)
                self.assertIsNotNone(final)
                self.assertGreater(j.cache_hits,0)
                self.assertEqual(len(paid),model.llm_counter.call_count)
                self.assertEqual(len(server.envs),2)
                store.close()

    def test_invalid_raw_reply_is_replayed_without_resampling_same_event(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'j.sqlite');clock=Clock();gate=Gate(store,clock.time,clock.sleep)
            store.put('circuit',{'phase':'READY'});tid=store.trial({'case':'invalid-json'});calls=[]
            def post(*args,**kwargs):
                raw='not JSON' if not calls else '{"index":0}'
                calls.append(raw)
                return response({'choices':[{'message':{'content':raw},'finish_reason':'stop'}]})
            for _ in range(2):
                j=Journal(store,tid,store.attempt(tid));llm=DurableLLM('fake',j,gate,post=post)
                self.assertEqual(llm.send_message('same'), 'not JSON')
                self.assertEqual(llm.send_message('same'), '{"index":0}')
            self.assertEqual(len(calls),2);store.close()


if __name__=='__main__':unittest.main()
