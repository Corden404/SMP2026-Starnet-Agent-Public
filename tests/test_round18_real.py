"""Paid stages remain gated; transport metadata, uncertainty and model boundaries."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round18_real as r

class RealTests(unittest.TestCase):
    def response(self,status=200,model='glm-4.6v',credit=False):
        body={'error':{'code':'1113','message':'insufficient credit'}} if credit else dict(model=model,choices=[dict(message={'content':'{"ok":true}'},finish_reason='stop')],usage={})
        return SimpleNamespace(status_code=status,text=json.dumps(body),json=lambda:body)
    def test_gate_before_any_key_or_transport(self):
        with patch.object(r,'require_local',side_effect=RuntimeError('gate')),patch.object(r,'HTTPXLLM') as llm:
            with self.assertRaises(RuntimeError):r.choose_model()
            llm.assert_not_called()
    def test_fenced_probe_accepted_without_new_request(self):
        self.assertTrue(r.parse_probe('```json\n{"ok":true}\n```'))
    def test_success_logs_no_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log.jsonl'
            fake=SimpleNamespace(post=lambda *a,**k:self.response(),close=lambda:None)
            with patch('round11_api.key',return_value='test-only-secret'),patch.object(r.transport,'client',return_value=fake):
                llm=r.HTTPXLLM('glm-4.6v',path);self.assertEqual(llm.send_message('hello'),'{"ok":true}');llm.close()
            text=path.read_text(encoding='utf8');self.assertNotIn('test-only-secret',text);self.assertNotIn('Authorization',text)
            self.assertEqual(llm.http_request_count,1)
    def test_credit_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            calls=[]
            fake=SimpleNamespace(post=lambda *a,**k:(calls.append(1) or self.response(429,credit=True)),close=lambda:None)
            with patch('round11_api.key',return_value='test'),patch.object(r.transport,'client',return_value=fake):
                llm=r.HTTPXLLM('glm-4.6v',Path(directory)/'log.jsonl')
                with self.assertRaises(r.APICreditExhausted):llm.send_message('hello')
                llm.close()
            self.assertEqual(calls,[1])
    def test_model_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fake=SimpleNamespace(post=lambda *a,**k:self.response(model='other'),close=lambda:None)
            with patch('round11_api.key',return_value='test'),patch.object(r.transport,'client',return_value=fake):
                llm=r.HTTPXLLM('glm-4.6v',Path(directory)/'log.jsonl')
                with self.assertRaises(r.ServiceAbort):llm.send_message('hello')
                llm.close()
    def test_malformed_payload_fails_outside_chain_retries(self):
        for body in ([],{'model':'glm-4.6v','choices':[None]}, {'model':'glm-4.6v','choices':[{'message':None}]}):
            with tempfile.TemporaryDirectory() as directory:
                response=SimpleNamespace(status_code=200,text=json.dumps(body),json=lambda:body)
                fake=SimpleNamespace(post=lambda *args,**kwargs:response,close=lambda:None)
                with patch('round11_api.key',return_value='test'),patch.object(r.transport,'client',return_value=fake):
                    llm=r.HTTPXLLM('glm-4.6v',Path(directory)/'log.jsonl')
                    with self.assertRaises(r.ServiceAbort):llm.send_message('hello')
                    self.assertEqual(llm.http_request_count,1);llm.close()
    def test_transport_uncertainty_persisted_and_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log.jsonl';calls=[]
            def post(*a,**k):calls.append(1);raise r.httpx.ConnectError('controlled')
            fake=SimpleNamespace(post=post,close=lambda:None)
            with patch('round11_api.key',return_value='test'),patch.object(r.transport,'client',return_value=fake):
                llm=r.HTTPXLLM('glm-4.6v',path)
                with self.assertRaises(r.ServiceAbort):llm.send_message('hello')
                llm.close()
            rows=[json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]
            self.assertEqual(rows[-1]['event'],'uncertain');self.assertTrue(rows[-1]['may_be_billed']);self.assertEqual(calls,[1])

if __name__=='__main__':unittest.main()
