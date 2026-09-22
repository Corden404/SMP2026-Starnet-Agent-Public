"""Scheduling-only resume tests; no real child processes, network or model calls."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round18_state as s
import round18_execute as x
import round18_closeout as close

class ExecuteTests(unittest.TestCase):
    def test_completed_fixed_block_not_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);entries=[{'spec':{'case':'r18-dev-000'}}]
            for v in ('PC2','Cut1'):s.write(out/'runs/dev'/('r18-dev-000-'+v+'.json'),{'status':'PASS'})
            s.write(out/'summaries/dev.json',{'status':'PASS'})
            with patch.object(x.e,'OUT',out),patch.object(x.e,'check',return_value={'seeds':{'dev':entries}}),\
                 patch.object(x.e,'variants',return_value=['PC2','Cut1']),patch.object(x,'child') as child:
                x.complete_pool('dev',6);child.assert_not_called()
    def test_missing_block_keeps_fixed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);entries=[{'spec':{'case':'r18-dev-000'}}]
            s.write(out/'runs/dev/r18-dev-000-PC2.json',{'status':'PASS'})
            with patch.object(x.e,'OUT',out),patch.object(x.e,'check',return_value={'seeds':{'dev':entries}}),\
                 patch.object(x.e,'variants',return_value=['PC2','Cut1']),patch.object(x,'child') as child:
                x.complete_pool('dev',6)
                self.assertEqual(child.call_args_list[0].args,('round18_study.py','run','--pool','dev','--start',0,'--stop',1))
                self.assertEqual(s.read(out/'runs/dev/r18-dev-000-PC2.json'),{'status':'PASS'})
    def test_incomplete_result_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch.object(s,'OUT',out),patch.object(s,'preserved',return_value={}),\
                 patch.object(x.e,'check',return_value={}),patch.object(x.resume,'inventory',return_value={'incomplete':[{'status':'STARTED'}]}),\
                 patch.object(x,'complete_pool') as run:
                with self.assertRaises(RuntimeError):x.resume_plan()
                run.assert_not_called()
    def test_no_selected_candidate_never_enters_paid_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);study=out/'study';cl=out/'closeout'
            s.write(study/'development-decision.json',{'selected':None});s.write(out/'progress.json',{'status':'COMPLETE_KEEP_PC2'})
            with patch.object(s,'OUT',out),patch.object(x.e,'OUT',study),patch.object(close,'OUT',cl),\
                 patch.object(s,'preserved',return_value={}),patch.object(x.e,'check',return_value={}),\
                 patch.object(x.resume,'inventory',return_value={'incomplete':[]}),patch.object(x,'complete_pool') as pool,\
                 patch.object(x,'child') as child:
                result=x.resume_plan(allow_paid=True)
                self.assertEqual([c.args[0] for c in pool.call_args_list],['dev','integration'])
                self.assertEqual([c.args[0] for c in child.call_args_list],['round18_closeout.py','round18_closeout.py'])
                self.assertEqual(result['selected'],None)

if __name__=='__main__':unittest.main()
