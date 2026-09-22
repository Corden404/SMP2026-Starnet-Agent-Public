import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round27_real as r

class RealTests(unittest.TestCase):
    def response(self,model='glm-4.6v',status=200,credit=False):
        body={'error':{'code':'1113','message':'insufficient credit'}} if credit else dict(model=model,choices=[dict(message={'content':'{"ok":true}'})])
        return SimpleNamespace(status_code=status,text=json.dumps(body),json=lambda:body)
    def test_model_validation_before_key(self):
        with patch.object(r,'key') as k:
            with self.assertRaises(ValueError):r.HTTPXLLM('unapproved','unused')
            k.assert_not_called()
    def test_primary_fallback_configuration(self):
        for model in r.MODELS:self.assertEqual(r.config(model)['model'],model)
    def test_one_call_no_secret_in_journal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'api.jsonl';fake=SimpleNamespace(post=lambda *a,**k:self.response(),close=lambda:None)
            with patch.object(r,'key',return_value='test-secret'),patch.object(r.transport,'client',return_value=fake):
                llm=r.HTTPXLLM('glm-4.6v',p);self.assertTrue(r.parse_probe(llm.send_message('health',True)));llm.close()
            self.assertEqual(llm.http_request_count,1);self.assertNotIn('test-secret',p.read_text());self.assertNotIn('Authorization',p.read_text())
    def test_credit_error_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            fake=SimpleNamespace(post=lambda *a,**k:self.response(status=429,credit=True),close=lambda:None)
            with patch.object(r,'key',return_value='test'),patch.object(r.transport,'client',return_value=fake):
                llm=r.HTTPXLLM('glm-4.6v',Path(d)/'api.jsonl')
                with self.assertRaises(r.APICreditExhausted):llm.send_message('health')
                self.assertEqual(llm.http_request_count,1);llm.close()
    def test_model_mismatch_aborts(self):
        with tempfile.TemporaryDirectory() as d:
            fake=SimpleNamespace(post=lambda *a,**k:self.response(model='other'),close=lambda:None)
            with patch.object(r,'key',return_value='test'),patch.object(r.transport,'client',return_value=fake):
                llm=r.HTTPXLLM('glm-4.6v',Path(d)/'api.jsonl')
                with self.assertRaises(r.ServiceAbort):llm.send_message('health')
                self.assertEqual(llm.http_request_count,1);llm.close()
    def test_tls_uncertainty_logged_once(self):
        import httpx
        with tempfile.TemporaryDirectory() as d:
            def fail(*a,**k):raise httpx.ConnectError('controlled')
            fake=SimpleNamespace(post=fail,close=lambda:None);p=Path(d)/'api.jsonl'
            with patch.object(r,'key',return_value='test'),patch.object(r.transport,'client',return_value=fake):
                llm=r.HTTPXLLM('glm-4.6v',p)
                with self.assertRaises(r.ServiceAbort):llm.send_message('health')
                self.assertEqual(llm.http_request_count,1);llm.close()
            self.assertEqual(json.loads(p.read_text().splitlines()[-1])['event'],'uncertain')

if __name__=='__main__':unittest.main()
