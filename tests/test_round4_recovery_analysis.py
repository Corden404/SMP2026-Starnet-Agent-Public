import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import test_round4_analysis as originals
import analyze_round4 as analysis

class RecoveryAnalysisTests(unittest.TestCase):
    def test_only_predeclared_matching_infrastructure_is_resolved(self):
        make=originals.Round4AnalysisTests().row
        a=make('a','B0',10);b=make('a','R1',11);b['attempt']=1
        failed=copy.deepcopy(b)
        failed.update(status='failed',attempt=0,error_type='HTTPError',error_http_status=502,
                      retryable_infrastructure=True)
        result=analysis.paired([a,b,failed],'R1',['a'],bootstrap=100)
        self.assertEqual(len(result['failures']),1)
        self.assertEqual(result['unresolved_failures'],[])
        for change in ({'error_http_status':401},{'retryable_infrastructure':False},
                       {'identity':{'definition':{}}}):
            fault=copy.deepcopy(failed);fault.update(change)
            result=analysis.paired([a,b,fault],'R1',['a'],bootstrap=100)
            self.assertEqual(len(result['unresolved_failures']),1)

    def test_duplicate_successes_are_rejected(self):
        make=originals.Round4AnalysisTests().row
        with self.assertRaisesRegex(ValueError,'Duplicate completed'):
            analysis.paired([make('a','B0',10),make('a','R1',11),make('a','R1',12)],'R1',['a'])

if __name__=='__main__':unittest.main()
