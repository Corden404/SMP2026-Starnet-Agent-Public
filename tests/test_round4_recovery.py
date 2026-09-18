import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_offline_baseline import PROJECT_ROOT
from round4_common import identity,save
import remote_round4_batch as module


class RecoveryTests(unittest.TestCase):
    def exercise(self,failures,error='ReadTimeout',status=None):
        class Process:
            def __init__(self,code):self.code=code
            def poll(self):return self.code
            def wait(self,timeout=None):return self.code
            def terminate(self):self.code=-1
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);calls=[]
            expected=identity({}, {}, {}, 'test', {})
            def launch(cmd,**kwargs):
                attempt=int(cmd[cmd.index('--attempt')+1]);calls.append(attempt)
                bad=len(calls)<=failures
                row={'case':'a','variant':'B0','repeat':0,'attempt':attempt,'identity':expected,
                     'status':'failed' if bad else 'completed'}
                if bad:row.update(error_type=error,error_http_status=status)
                else:row['final_score']=7
                save(folder/'test'/f'a-B0-r0-a{attempt}.json',row)
                return Process(1 if bad else 0)
            with patch.object(module,'OUT',folder),patch.object(module.subprocess,'Popen',side_effect=launch),\
                 patch.object(module.time,'sleep'):
                error_seen=None
                try:module.run_batch(['a'],['B0'],'test',remote=True,workers=1)
                except RuntimeError as exc:error_seen=str(exc)
            rows=[json.loads(p.read_text(encoding='utf-8')) for p in (folder/'test').glob('*-a*.json')]
            return calls,rows,error_seen

    def test_transient_restarts_new_session_and_preserves_failed_attempt(self):
        calls,rows,error=self.exercise(1)
        self.assertIsNone(error);self.assertEqual(calls,[0,1])
        self.assertEqual(sorted(r['status'] for r in rows),['completed','failed'])

    def test_retry_limit_is_finite_and_auth_is_not_retried(self):
        calls,rows,error=self.exercise(9)
        self.assertEqual(calls,[0,1,2]);self.assertIn('retry limit',error)
        calls,rows,error=self.exercise(9,'HTTPError',401)
        self.assertEqual(calls,[0]);self.assertIsNotNone(error)


if __name__=='__main__':unittest.main()
