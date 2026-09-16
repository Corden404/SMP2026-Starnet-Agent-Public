import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import benchmark

class BenchmarkContractTests(unittest.TestCase):
    def test_evaluation_consumes_session_without_losing_results(self):
        class ConsumingEnv:
            def __init__(self,*args,**kwargs):self.ended=False
            def get_remaining_budget(self):
                if self.ended:raise AssertionError('No queries allowed after evaluate')
                return 10.0
            def trigger_eval(self):
                if self.ended:raise AssertionError('Evaluation must run exactly once')
                self.ended=True
                return 12.5
        class Model:
            def __init__(self,*args):
                self.llm_counter=SimpleNamespace(call_count=0)
                self.stop_reason='test complete'
            def step(self):return 1
        seed={'global_setting':{'max_budget':10,'max_api_calls':20},
              'nodes':[{'id':1}],'edges':[],'prompts':{'1':15}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'workspace/team_submission'
            source.mkdir(parents=True)
            (source/'config.json').write_text(json.dumps({'person':[{'runtime':{}},{},{}]}),encoding='utf-8')
            (source/'starnet_model.py').write_text('# fixture only',encoding='utf-8')
            with patch.object(benchmark,'ROOT',root),patch.object(benchmark,'make_seed',return_value=seed),\
                 patch.object(benchmark,'RemoteStarNetEnv',ConsumingEnv),\
                 patch.object(benchmark,'ParticipantSquadModel',Model),\
                 patch.object(benchmark,'ZhipuLLM',return_value=SimpleNamespace(http_request_count=0)),\
                 patch.dict(benchmark.os.environ,{'ZHIPUAI_API_KEY':'offline-placeholder'}),\
                 contextlib.redirect_stdout(io.StringIO()):
                benchmark.run('official','score')
            result=json.loads((root/'agent_notes/experiments/round2/official-score.json').read_text(encoding='utf-8'))
            self.assertEqual(result['final_score'],12.5)
            self.assertEqual(result['remaining_budget'],10.0)
            self.assertEqual(result['steps'],1)

if __name__=='__main__':unittest.main()
