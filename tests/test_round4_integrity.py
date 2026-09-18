import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_offline_baseline import FakeLLM, StrictFakeEnv, person_list
from starnet_model import ParticipantSquadModel
from round4_common import OUT, TraceLLM, attach_traces, cached, freeze_baseline, identity, save
from zhipu import ZhipuLLM


class Round4IntegrityTests(unittest.TestCase):
    def test_identity_changes_and_incomplete_cache_rejected(self):
        config = {'endpoint': 'https://example.test/chat', 'model': 'A', 'temperature': .3}
        def make(seed=None, model=None, runtime=None, repeat=0):
            return identity(seed or {'nodes': [1]}, runtime or {'strategy': 'B0'}, model or config,
                            'test', {'type': 'proxy'}, repeat=repeat)
        original = make()
        variants = [make(seed={'nodes':[2]}), make(model={**config,'model':'B'}),
                    make(model={**config,'temperature':.2}), make(runtime={'strategy':'R1'}), make(repeat=1)]
        with patch('round4_common.file_hash', return_value='changed'):
            variants.append(make())
        self.assertTrue(all(v['run_id'] != original['run_id'] for v in variants))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'run.json'
            save(p, {'identity':original, 'status':'completed', 'final_score':1})
            self.assertEqual(cached(p, original)['final_score'], 1)
            for other in variants:
                with self.assertRaises(RuntimeError): cached(p, other)
            save(p, {'identity':original,'status':'running','final_score':1})
            with self.assertRaises(RuntimeError): cached(p, original)
        with self.assertRaises(ValueError): make(model={**config,'api_key':'secret'})
        with self.assertRaises(ValueError): make(model={**config,'endpoint':'https://user:pass@host/chat'})

    def test_frozen_baseline_bridge_preserves_actions_and_calls(self):
        freeze_baseline()
        spec = importlib.util.spec_from_file_location('frozen_round4_baseline', OUT/'baseline/starnet_model.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for n,budget,limit in [(4,60,50),(50,100,120),(100,200,250)]:
            people = person_list(n,limit,limit)
            people[0]['runtime'].update(strategy='planner',planner={'decision_mode':'roi','roi_floor':1.0})
            results=[]
            for model_class in [module.ParticipantSquadModel,ParticipantSquadModel]:
                env=StrictFakeEnv(n,budget)
                llm=TraceLLM(FakeLLM(['bad','bad','bad','{"targets":[]}', '{"index":999}'] + ['{"index":0}']*250))
                with contextlib.redirect_stdout(io.StringIO()):
                    m=model_class(env,copy.deepcopy(people),llm)
                    events=attach_traces(m,llm)
                    for step in range(limit):
                        if m.step():break
                    else:self.fail('fuse reached')
                results.append((env._actions,env._budget,m.llm_counter.call_count,m.stop_reason))
            self.assertEqual(results[0],results[1])
            origins={e['decision_origin'] for e in events}
            self.assertIn('model_valid',origins)
            self.assertTrue(origins & {'invalid_index_fallback','invalid_targets_fallback'})

    def test_default_adapter_payload_unchanged_json_is_explicit(self):
        class Response:
            status_code=200
            text=json.dumps({'id':'test','model':'reported-model','choices':[{'message':{'content':'{}'},'finish_reason':'stop'}]})
        for enabled in [False,True]:
            m=ZhipuLLM('fake-key',5,{'json_mode':enabled})
            with patch('zhipu.requests.post',return_value=Response()) as post:
                m.send_message('JSON',json_flag=True)
            data=post.call_args.kwargs['json']
            self.assertEqual(data['model'],'glm-4-plus')
            self.assertEqual('response_format' in data, enabled)
            self.assertEqual(m.response_metadata[0]['model'],'reported-model')
            self.assertEqual(m.http_request_count,1)


if __name__ == '__main__': unittest.main()
