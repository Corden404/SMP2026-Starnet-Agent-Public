import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_offline_baseline import PROJECT_ROOT
import recovery_runner as runner
from recovery_store import Store
from recovery_analysis import summarize


class RecoverySupervisorTests(unittest.TestCase):
    def test_hard_watchdog_stops_child_and_persists_global_stop(self):
        class Process:
            def __init__(self):self.returncode=None;self.terminated=False
            def poll(self):return self.returncode
            def terminate(self):self.returncode=-1;self.terminated=True
            def wait(self,timeout):return self.returncode
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=Store(root/'journal.sqlite')
            store.put('circuit',{'phase':'RECOVERING','deadline':1000});store.close();proc=Process()
            with patch.object(runner,'RECOVERY',root),patch.object(runner.subprocess,'Popen',return_value=proc),\
                 patch.object(runner.time,'time',return_value=1001),patch.object(runner,'inventory',return_value={}),\
                 patch.object(runner,'report',return_value={}):
                self.assertEqual(runner.supervise(),2)
            self.assertTrue(proc.terminated)
            self.assertEqual(json.loads((root/'watchdog-stop.json').read_text())['reason'],'THREE_HOUR_OUTAGE')
            store=Store(root/'journal.sqlite')
            self.assertEqual(store.get('circuit')['phase'],'BLOCKED');store.close()

    def test_missing_or_infrastructure_failed_cohort_has_no_summary_mean(self):
        rows=[{'status':'COMPLETED','kind':'development','case':'a','variant':'B0','point':None,'label':None,
               'existing_valid':[],'result':{'final_score':10}},
              {'status':'INFRA_FAILURE','kind':'development','case':'a','variant':'R1','point':None,'label':None,
               'existing_valid':[],'result':{'final_score':999}}]
        result=summarize(rows)
        self.assertEqual(result['development_status'],'INCOMPLETE')
        self.assertNotIn('development',result)


if __name__=='__main__':unittest.main()
