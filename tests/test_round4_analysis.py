import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_offline_baseline import PROJECT_ROOT
import remote_round4_batch

import analyze_round4 as module


class Round4AnalysisTests(unittest.TestCase):
    def row(self,case,variant,score,repeat=0):
        return {'case':case,'variant':variant,'repeat':repeat,'status':'completed','final_score':score,
                'spec':{'family':'ba','scale':1,'mode':'independent','persona_mix':0},
                'natural_score':0,'steps':80,'llm_calls':35,'elapsed_seconds':1,
                'evaluation':{'type':'test'},'identity':{'definition':{k:'same' for k in
                    ('seed_sha256','model_config','submission_files','harness_files','environment','packages')}}}

    def test_repeats_are_aggregated_per_seed_and_failures_retained(self):
        rows=[self.row('a','B0',10),self.row('a','B0',20,1),self.row('a','R1',20),self.row('a','R1',30,1),
              self.row('b','B0',20),self.row('b','R1',15)]
        s=module.paired(rows,'R1',['a','b'],bootstrap=100)
        self.assertEqual(s['n'],2);self.assertEqual(s['mean_delta'],2.5)
        rows.append({'case':'a','variant':'R1','status':'failed','attempt':1})
        self.assertEqual(len(module.paired(rows,'R1',['a','b'],bootstrap=100)['failures']),1)
        rows[2]['identity']['definition']['model_config']='different'
        with self.assertRaises(ValueError):module.paired(rows,'R1',['a','b'])

    def test_empty_credit_stops_every_owned_worker(self):
        class Process:
            def __init__(self,code):self.code=code;self.terminated=False
            def poll(self):return self.code
            def terminate(self):self.terminated=True;self.code=-1
            def wait(self,timeout=None):return self.code
        processes=[Process(13),Process(None),Process(None)]
        with tempfile.TemporaryDirectory() as directory,patch.object(remote_round4_batch,'OUT',Path(directory)),\
             patch.object(remote_round4_batch.subprocess,'Popen',side_effect=processes) as popen:
            with self.assertRaisesRegex(RuntimeError,'exit 13'):
                remote_round4_batch.run_batch(['a','b','c','d'],['B0'],'test',workers=3)
            self.assertEqual(popen.call_count,3)
            self.assertTrue(processes[1].terminated and processes[2].terminated)


if __name__=='__main__':unittest.main()
