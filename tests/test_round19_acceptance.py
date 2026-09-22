"""Conditional-stage and safe-resumption checks; never starts a strategy or API cohort."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round19_state as s
import round19_resume as resume
import round19_closeout as close

class AcceptanceTests(unittest.TestCase):
    def test_failed_diagnostic_requires_no_new_cohorts(self):
        self.assertEqual(close.required_study({'decision':{'eligible_ct1':False}}),{})
    def test_successful_diagnostic_needs_development(self):
        self.assertEqual(close.required_study({'decision':{'eligible_ct1':True}}),{'development':'PENDING'})
    def test_no_candidate_stops_after_development(self):
        got=close.required_study({'decision':{'eligible_ct1':True}},{'variants':['PC2','Cut1','CT1'],'selected':None})
        self.assertEqual(got,{'dev':162,'integration':18})
    def test_selected_candidate_requires_fixed_confirmation(self):
        got=close.required_study({'decision':{'eligible_ct1':True}},{'variants':['PC2','Cut1','CT1','CT1R'],'selected':'CT1R'},{'eligible_real':False})
        self.assertEqual(got,{'dev':216,'integration':24,'confirm':432,'stress':108})
    def test_paid_cohort_only_after_local_gate(self):
        got=close.required_study({'decision':{'eligible_ct1':True}},{'variants':['PC2','Cut1','CT1'],'selected':'CT1'},{'eligible_real':True})
        self.assertEqual(got['real'],96)
    def test_recovery_fences_off_paid_old_and_protocol_files(self):
        for p in ('../round18/progress.json','real/runs/a.json','diagnostic/protocol.json','../../outside.json'):
            with self.assertRaises(ValueError):resume.target(p)
    def test_recovery_refuses_successful_records(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'round19';p=out/'diagnostic/states/a.json'
            s.write(p,{'status':'PASS'},True);before=s.sha(p)
            with patch.object(s,'OUT',out):
                with self.assertRaises(RuntimeError):resume.recover('diagnostic/states/a.json','test')
            self.assertEqual(s.sha(p),before)
    def test_recovery_reason_required(self):
        with self.assertRaises(ValueError):resume.recover('diagnostic/states/a.json',' ')
    def test_record_atomic_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'record.json';s.write(p,{'phase':'STARTED'});s.write(p,{'phase':'PASS'})
            self.assertEqual(s.read(p),{'phase':'PASS'});self.assertEqual(list(p.parent.glob('*.tmp')),[])

if __name__=='__main__':unittest.main()
