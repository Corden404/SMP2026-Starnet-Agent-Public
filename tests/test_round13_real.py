"""Exact API fallback selection tested with mocks only, never personal credentials."""
from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round13_real as real

class RealSelectionTests(unittest.TestCase):
    def run_selection(self,side_effect):
        with tempfile.TemporaryDirectory() as d,contextlib.ExitStack() as s:
            s.enter_context(patch.object(real.r,'OUT',Path(d)))
            s.enter_context(patch.object(real.r,'check',return_value={}))
            real.r.save(Path(d)/'local-decision.json',{'eligible_for_real':True})
            s.enter_context(patch.object(real,'last_error',return_value={'code':'1113'}))
            probe=s.enter_context(patch.object(real,'probe',side_effect=side_effect))
            s.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result=real.select()
            return result,[c.args[0] for c in probe.call_args_list]
    def test_credit_unlocks_exact_air_not_45air(self):
        def fake(model,out):
            if model=='glm-4-plus':raise real.APICreditExhausted()
            return {'http_calls':1,'config':{'model':model}}
        result,calls=self.run_selection(fake)
        self.assertEqual(calls,['glm-4-plus','glm-4-air']);self.assertEqual(result['model'],'glm-4-air')
    def test_all_three_credit_failures_preserved(self):
        result,calls=self.run_selection(real.APICreditExhausted())
        self.assertEqual(calls,list(real.MODELS));self.assertEqual(result['status'],'NO_AUTHORIZED_MODEL_AVAILABLE')
        self.assertEqual(len(result['attempts']),3)
    def test_uncertain_service_does_not_unlock_fallback(self):
        result,calls=self.run_selection(real.ServiceAbort('uncertain'))
        self.assertEqual(calls,['glm-4-plus']);self.assertEqual(result['status'],'SERVICE_BLOCKED')
    def test_fenced_success_reuses_receipt_without_request(self):
        import json
        with tempfile.TemporaryDirectory() as d,patch.object(real.r,'OUT',Path(d)),patch.object(real,'probe') as probe,contextlib.redirect_stdout(io.StringIO()):
            real.r.save(Path(d)/'api/selection.json',{'status':'SERVICE_BLOCKED','attempts':[{'model':'glm-4.6v'}]})
            response={'event':'response','status':200,'body':json.dumps({'model':'glm-4.6v','choices':[{'finish_reason':'stop','message':{'content':'```json\n{\"ok\":true}\n```'}}]})}
            (Path(d)/'api/probe-glm-4.6v.jsonl').write_text(json.dumps(response)+'\n',encoding='utf8')
            result=real.review_successful_probe()
            self.assertEqual(result['status'],'SELECTED');self.assertEqual(result['new_requests'],0)
            probe.assert_not_called()

    def test_healthy_primary_no_extra_model_probe(self):
        result,calls=self.run_selection(lambda model,out: {'http_calls':1,'config':{'model':model}})
        self.assertEqual(calls,['glm-4-plus']);self.assertEqual(result['status'],'SELECTED')

if __name__=='__main__':unittest.main()
