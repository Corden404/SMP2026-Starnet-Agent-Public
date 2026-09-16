import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_offline_baseline import PROJECT_ROOT
from analyze_round3 import best_attempts, paired_stats
import remote_round3_batch

class ExperimentIntegrityTests(unittest.TestCase):
    def test_uses_first_successful_attempt_not_highest_score(self):
        rows=[{'case':'x','variant':'score','repeat':2,'final_score':1000,'technical_failure':False},
              {'case':'x','variant':'score','repeat':0,'final_score':-10,'technical_failure':True},
              {'case':'x','variant':'score','repeat':1,'final_score':20,'technical_failure':False}]
        self.assertEqual(best_attempts(rows)[('x','score')]['repeat'],1)

    def test_pairing_reports_actual_differences_and_resource_maxima(self):
        def row(case,score,steps):
            return {'case':case,'final_score':score,'steps':steps,'llm_calls':steps//2,
                    'natural_score':0,'elapsed_seconds':1,'repeat':0}
        stats=paired_stats([(row('a',10,10),row('a',20,12)),(row('b',30,10),row('b',25,14))])
        self.assertEqual(stats['paired_delta'],2.5)
        self.assertEqual((stats['wins'],stats['losses']),(1,1))
        self.assertEqual(stats['max_steps'],14)

    def test_empty_credit_worker_stops_scheduling_and_terminates_siblings(self):
        class Process:
            def __init__(self,status,pid):self.status=status;self.pid=pid;self.terminated=False
            def poll(self):return self.status
            def terminate(self):self.terminated=True;self.status=-1
            def wait(self,timeout=None):return self.status
        processes=[Process(13,101),Process(None,102),Process(None,103)]
        with tempfile.TemporaryDirectory() as directory,\
             patch.object(remote_round3_batch.sys,'argv',['batch','--cases','a','b','c','d','--variants','score','--output',directory]),\
             patch.object(remote_round3_batch.subprocess,'Popen',side_effect=processes) as popen,\
             patch.object(remote_round3_batch.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'exit 13'):remote_round3_batch.main()
            self.assertEqual(popen.call_count,3)
            self.assertTrue(processes[1].terminated and processes[2].terminated)

if __name__=='__main__':unittest.main()
