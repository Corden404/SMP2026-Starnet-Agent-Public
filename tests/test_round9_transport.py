"""Offline checks for paid-call journaling and fail-closed service handling."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import requests

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
from round9_api import JournalLLM, ServiceAbort
from zhipu import APICreditExhausted


class Response:
    def __init__(self,status,payload):
        self.status_code=status;self.payload=payload;self.text=json.dumps(payload)
    def json(self):return self.payload


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'journal.jsonl'
        self.key_patch=patch('round9_api.key',return_value='test-placeholder-not-a-credential')
        self.key_patch.start()
    def tearDown(self):
        self.key_patch.stop();self.temp.cleanup()
    def test_credit_error_stops_without_retry(self):
        llm=JournalLLM('glm-4.5-air',self.path)
        with patch('round9_api.requests.post',return_value=Response(429,{'error':{'code':'1113','message':'insufficient balance'}})) as post:
            with self.assertRaises(APICreditExhausted):llm._post_once('https://example.invalid',{'model':'glm-4.5-air'})
        self.assertEqual(post.call_count,1);self.assertEqual(llm.http_request_count,1)
    def test_http_error_bypasses_casevo_retry(self):
        llm=JournalLLM('glm-4.5-air',self.path)
        with patch('round9_api.requests.post',return_value=Response(502,{'error':'gateway'})) as post:
            with self.assertRaises(ServiceAbort):llm._post_once('https://example.invalid',{})
        self.assertEqual(post.call_count,1)
        self.assertTrue(issubclass(ServiceAbort,SystemExit))
    def test_uncertain_outcome_preserved_and_not_retried(self):
        llm=JournalLLM('glm-4.5-air',self.path)
        with patch('round9_api.requests.post',side_effect=requests.ConnectionError('test connection loss')) as post:
            with self.assertRaises(ServiceAbort):llm._post_once('https://example.invalid',{})
        events=[json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(post.call_count,1)
        self.assertEqual(events[-1]['event'],'uncertain')
        self.assertTrue(events[-1]['may_be_billed'])
    def test_global_stop_prevents_new_request(self):
        stop=Path(self.temp.name)/'STOP.json';stop.write_text('{}')
        llm=JournalLLM('glm-4.5-air',self.path,stop)
        with patch('round9_api.requests.post') as post:
            with self.assertRaises(ServiceAbort):llm._post_once('https://example.invalid',{})
        post.assert_not_called();self.assertFalse(self.path.exists())
    def test_journal_never_contains_authorization_header(self):
        llm=JournalLLM('glm-4.5-air',self.path)
        with patch('round9_api.requests.post',return_value=Response(200,{'ok':True})):
            llm._post_once('https://example.invalid',{'model':'glm-4.5-air'})
        text=self.path.read_text(encoding='utf-8')
        self.assertNotIn('Authorization',text);self.assertNotIn('test-placeholder',text)
    def test_physical_fuse_and_model_allowlist(self):
        llm=JournalLLM('glm-4.6v',self.path);llm.http_request_count=119
        with patch('round9_api.requests.post') as post:
            with self.assertRaises(ServiceAbort):llm._post_once('https://example.invalid',{})
        post.assert_not_called()
        with self.assertRaises(ValueError):JournalLLM('unapproved-model',self.path)

if __name__=='__main__':unittest.main()
