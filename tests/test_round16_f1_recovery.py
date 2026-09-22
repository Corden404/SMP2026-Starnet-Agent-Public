"""No-network strict F1 replay and extra-attempt accounting tests."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round16_f1_recovery as r

class RecoveryTests(unittest.TestCase):
    def response(self):
        return dict(event='response',number=1,status=200,body=json.dumps({'model':r.MODEL,'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}))
    def test_completed_response_is_reused_no_key_no_network(self):
        response=self.response()
        with patch.object(r,'key') as key,patch.object(r.requests,'post') as post:
            llm=r.ReplayLLM('unused',[(dict(data={'x':1}),response)])
            actual=llm._post_once('unused',{'x':1})
            self.assertEqual(actual.text,response['body']);self.assertEqual(llm.http_request_count,1)
            key.assert_not_called();post.assert_not_called()
    def test_payload_mismatch_fails_before_any_request(self):
        with patch.object(r.requests,'post') as post:
            llm=r.ReplayLLM('unused',[(dict(data={'x':1}),self.response())])
            with self.assertRaises(r.RecoveryMismatch):llm._post_once('unused',{'x':2})
            post.assert_not_called()
    def test_uncertain_attempt_charged_once_to_both_counters(self):
        class Counter:
            call_count=23
            def _reserve_call(self):self.call_count+=1
        llm=r.ReplayLLM('unused',pending={'data':{'x':1}});llm.counter=Counter();llm.http_request_count=23
        with self.assertRaises(r.ReplayComplete):llm._post_once('unused',{'x':1})
        self.assertEqual(llm.http_request_count,24);self.assertEqual(llm.counter.call_count,24)
        self.assertEqual(llm.uncertain_charged,1)
        with self.assertRaises(r.ReplayComplete):llm._post_once('unused',{'x':1})
        self.assertEqual(llm.http_request_count,24)
    def test_uncertain_payload_mismatch_not_retried(self):
        llm=r.ReplayLLM('unused',pending={'data':{'x':1}})
        with self.assertRaises(r.RecoveryMismatch):llm._post_once('unused',{'x':2})
        self.assertEqual(llm.new_requests,0)
    def test_boundary_checks_actions_before_new_request(self):
        llm=r.ReplayLLM('unused')
        def deny():raise r.RecoveryMismatch('wrong prefix')
        llm.boundary=deny
        with self.assertRaises(r.RecoveryMismatch):llm._post_once('unused',{})
        self.assertEqual(llm.new_requests,0)
    def test_chain_matching_ignores_only_metadata(self):
        x=dict(role='scout',chain='batch',operation=0,input={'a':1},output={'targets':[1]},real_llm=True)
        y={k:v for k,v in x.items() if k!='real_llm'}
        self.assertTrue(r.same_chains([x],[y]))
        y['output']={'targets':[2]};self.assertFalse(r.same_chains([x],[y]))
    def test_journal_rejects_unknown_request_gap(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'j.jsonl';p.write_text(json.dumps(dict(event='request',number=1,data={}))+'\n',encoding='utf8')
            with self.assertRaises(r.RecoveryMismatch):r.replay_records(p)
    def test_journal_accepts_only_last_uncertain(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'j.jsonl';events=[dict(event='request',number=1,data={}),self.response(),dict(event='request',number=2,data={'x':1}),dict(event='uncertain',number=2)]
            p.write_text('\n'.join(json.dumps(x) for x in events),encoding='utf8')
            completed,pending=r.replay_records(p)
            self.assertEqual(len(completed),1);self.assertEqual(pending['number'],2)
    def test_original_failed_model_and_counters(self):
        for name,expected in [('r13-glm-000-S44',(23,True)),('r13-glm-001-F1',(27,False))]:
            complete,pending=r.replay_records(r.OLD_RUNS/(name+'.http.jsonl'))
            self.assertEqual((len(complete),pending is not None),expected)

if __name__=='__main__':unittest.main()
