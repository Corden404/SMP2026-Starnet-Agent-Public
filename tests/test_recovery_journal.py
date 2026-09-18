import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_offline_baseline import PROJECT_ROOT
import requests
from recovery_store import Store,Journal,RecoveryStop,InfrastructureFailure
from recovery_env import Gate,Transport,DurableEnv
from recovery_llm import DurableLLM
from team_submission.starnet_model import CountingLLM


def response(value,status=200):
    r=requests.Response();r.status_code=status;r._content=json.dumps(value).encode();r.headers={};return r


class Clock:
    def __init__(self):self.now=1000.
    def time(self):return self.now
    def sleep(self,x):self.now+=x


class Server:
    def __init__(self,fail_at=None,after_apply=True):
        self.sessions={};self.requests=[];self.fail_at=fail_at;self.after_apply=after_apply;self.fail_once=True;self.drift=False
    def post(self,url,json,timeout):
        endpoint=url.split('/api/')[-1];self.requests.append((endpoint,dict(json)))
        if endpoint=='start_session':
            sid='s'+str(len(self.sessions)+1);self.sessions[sid]={'budget':100,'calls':0};return response({'session_id':sid})
        state=self.sessions[json['session_id']]
        if endpoint=='get_budget':return response({'budget':state['budget']+(1 if self.drift else 0)})
        if endpoint=='communicate':
            if state['calls']+1==self.fail_at and self.fail_once:
                self.fail_once=False
                if self.after_apply:state['calls']+=1;state['budget']-=2
                raise requests.ReadTimeout('Lost response')
            state['calls']+=1;state['budget']-=2
            return response({'status':'success','new_w':state['calls']})
        if endpoint=='evaluate':
            if self.fail_at=='evaluate' and self.fail_once:self.fail_once=False;raise requests.ReadTimeout('Lost evaluation')
            return response({'final_score':state['calls']})
        raise AssertionError(endpoint)


class RecoveryJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'state.sqlite'
        self.store=Store(self.path);self.clock=Clock();self.gate=Gate(self.store,self.clock.time,self.clock.sleep)
        self.store.put('circuit',{'phase':'READY'});self.paid=0
    def tearDown(self):self.store.close();self.temp.cleanup()
    def provider(self,*args,**kwargs):
        self.paid+=1
        return response({'choices':[{'message':{'content':'{"index":0}'},'finish_reason':'stop'}],
                         'id':'r'+str(self.paid),'model':'glm-4-plus','usage':{'total_tokens':10}})
    def run_attempt(self,server,tid,crash=False):
        aid=self.store.attempt(tid);j=Journal(self.store,tid,aid)
        env=DurableEnv('https://sandbox.invalid',{},j,Transport(self.store,self.gate,aid,post=server.post))
        llm=DurableLLM('fake',j,self.gate,post=self.provider);count=CountingLLM(llm,120)
        for i in range(10):
            env.get_remaining_budget();llm.state=lambda i=i:{'step':i}
            self.assertEqual(count.send_message('decision '+str(i),True),'{"index":0}')
            if crash and i==6:raise RuntimeError('process crash before action')
            env.communicate(1,1)
        score=env.trigger_eval();j.assert_end();self.store.complete(tid,aid,{'score':score})
        return count.call_count,j.cache_hits

    def test_ten_calls_not_seventeen_after_unknown_applied_action(self):
        server=Server(7);tid=self.store.trial({'case':'a','repeat':0})
        with self.assertRaises(InfrastructureFailure):self.run_attempt(server,tid)
        self.assertEqual(self.paid,7)
        calls,hits=self.run_attempt(server,tid)
        self.assertEqual((self.paid,calls,hits),(10,10,7))
        self.assertEqual(len(server.sessions),2)
        self.assertEqual(server.sessions['s1']['calls'],7)
        self.assertEqual(self.store.execute("SELECT COUNT(*) FROM trials WHERE status='COMPLETED'").fetchone()[0],1)

    def test_unknown_unapplied_action_has_same_recovery(self):
        server=Server(7,after_apply=False);tid=self.store.trial({'case':'a'})
        with self.assertRaises(InfrastructureFailure):self.run_attempt(server,tid)
        self.run_attempt(server,tid);self.assertEqual(self.paid,10)
        self.assertEqual(server.sessions['s1']['calls'],6)

    def test_crash_after_durable_reply_recovers_across_store_reopen(self):
        server=Server();tid=self.store.trial({'case':'crash'})
        with self.assertRaises(RuntimeError):self.run_attempt(server,tid,True)
        self.store.close();self.store=Store(self.path);self.gate=Gate(self.store,self.clock.time,self.clock.sleep)
        self.run_attempt(server,tid);self.assertEqual(self.paid,10)

    def test_evaluation_failure_reuses_all_model_responses(self):
        server=Server('evaluate');tid=self.store.trial({'case':'eval'})
        with self.assertRaises(InfrastructureFailure):self.run_attempt(server,tid)
        _,hits=self.run_attempt(server,tid);self.assertEqual((self.paid,hits),(10,10))

    def test_changed_budget_stops_before_any_new_paid_call(self):
        server=Server(7);tid=self.store.trial({'case':'drift'})
        with self.assertRaises(InfrastructureFailure):self.run_attempt(server,tid)
        server.drift=True
        with self.assertRaisesRegex(RecoveryStop,'REPLAY_DIVERGENCE'):self.run_attempt(server,tid)
        self.assertEqual(self.paid,7)

    def test_request_state_and_position_must_all_match(self):
        tid=self.store.trial({'case':'x'});aid=self.store.attempt(tid)
        j=Journal(self.store,tid,aid);llm=DurableLLM('fake',j,self.gate,post=self.provider);llm.state=lambda:{'queue':[1,2]}
        llm.send_message('a');llm.send_message('a');self.assertEqual(self.paid,2)
        for prompt,state in [('b',{'queue':[1,2]}),('a',{'queue':[2,1]})]:
            j=Journal(self.store,tid,self.store.attempt(tid));llm=DurableLLM('fake',j,self.gate,post=self.provider);llm.state=lambda:state
            with self.assertRaisesRegex(RecoveryStop,'REPLAY_DIVERGENCE'):llm.send_message(prompt)
        self.assertEqual(self.paid,2)

    def test_model_and_independent_repeat_are_not_shared(self):
        for identity in [{'repeat':0,'model':'a'},{'repeat':1,'model':'a'},{'repeat':0,'model':'b'}]:
            tid=self.store.trial(identity);j=Journal(self.store,tid,self.store.attempt(tid))
            DurableLLM('fake',j,self.gate,post=self.provider).send_message('same')
        self.assertEqual(self.paid,3)

    def test_unknown_paid_response_does_not_retry(self):
        tid=self.store.trial({'case':'unknown'});j=Journal(self.store,tid,self.store.attempt(tid))
        def fail(*a,**k):self.paid+=1;raise requests.ReadTimeout()
        with self.assertRaisesRegex(RecoveryStop,'LLM_RESULT_UNKNOWN'):DurableLLM('fake',j,self.gate,post=fail).send_message('x')
        j=Journal(self.store,tid,self.store.attempt(tid))
        with self.assertRaisesRegex(RecoveryStop,'LLM_RESULT_UNKNOWN'):DurableLLM('fake',j,self.gate,post=self.provider).send_message('x')
        self.assertEqual(self.paid,1)

    def test_waiting_blocks_chat_and_embedding(self):
        self.gate.outage('502');tid=self.store.trial({'case':'blocked'})
        for method,arg in [('send_message','x'),('send_embedding',['x'])]:
            tid=self.store.trial({'blocked_method':method})
            j=Journal(self.store,tid,self.store.attempt(tid));llm=DurableLLM('fake',j,self.gate,post=self.provider)
            with self.assertRaisesRegex(RecoveryStop,'PAID_GATE_CLOSED'):getattr(llm,method)(arg)
        self.assertEqual(self.paid,0)

    def test_deadline_survives_short_success_and_three_flaps_stop(self):
        self.gate.outage('first');deadline=self.store.get('circuit')['deadline']
        for n in range(3):
            state=self.store.get('circuit');state['phase']='RECOVERING';self.store.put('circuit',state)
            self.clock.now+=10
            if n==2:
                with self.assertRaisesRegex(RecoveryStop,'PERSISTENT_SANDBOX_FAILURE'):self.gate.outage('502',trial_failure=True)
            else:self.gate.outage('502',trial_failure=True)
            self.assertEqual(self.store.get('circuit')['deadline'],deadline)

    def test_three_hour_deadline_and_completed_trial_reset(self):
        self.gate.outage('first');deadline=self.store.get('circuit')['deadline'];self.clock.now=deadline
        with self.assertRaisesRegex(RecoveryStop,'THREE_HOUR_OUTAGE'):self.gate.check(health=True)
        self.gate.completed();self.assertNotIn('deadline',self.store.get('circuit'))

    def test_three_budget_checks_do_not_bypass_failed_canary(self):
        calls={'probe':0,'canary':0}
        def probe():calls['probe']+=1
        def canary():
            calls['canary']+=1
            if calls['canary']==1:raise InfrastructureFailure('communicate 502')
        self.gate.recover(probe,canary)
        self.assertEqual(calls,{'probe':6,'canary':2});self.assertEqual(self.paid,0)
        self.assertEqual(self.store.get('circuit')['phase'],'RECOVERING')

    def test_duplicate_runner_and_failed_storage_prevent_requests(self):
        with self.assertRaisesRegex(RecoveryStop,'RUNNER_LOCKED'):Store(self.path)
        tid=self.store.trial({'case':'disk'});j=Journal(self.store,tid,self.store.attempt(tid));self.store.db.close()
        with self.assertRaisesRegex(RecoveryStop,'STORAGE_FAILURE'):DurableLLM('fake',j,self.gate,post=self.provider).send_message('x')
        self.assertEqual(self.paid,0)

    def test_recovery_signal_is_not_caught_as_strategy_exception(self):
        caught=False
        try:
            try:raise InfrastructureFailure('502')
            except Exception:caught=True
        except InfrastructureFailure:pass
        self.assertFalse(caught)


if __name__=='__main__':unittest.main()
