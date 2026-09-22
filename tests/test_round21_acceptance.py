import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round21_state as s
class Acceptance(unittest.TestCase):
    def test_local_gate_blocks_real(self):
        x=s.read(s.OUT/'study/local-decision.json');self.assertFalse(x['eligible_real']);self.assertFalse(x['confirmed_local'])
        self.assertGreater(x['confirm_mean'],0);self.assertGreaterEqual(x['stress_mean'],0)
    def test_progress_marks_real_skipped(self):
        p=s.read(s.OUT/'progress.json');self.assertEqual(p['steps']['R4']['status'],'COMPLETE');self.assertEqual(p['steps']['R5']['status'],'SKIPPED_GATE')
    def test_all_fixed_cohorts_complete(self):
        expected={'dev':162,'integration':18,'confirm':432,'stress':108}
        for pool,n in expected.items():
            rows=list((s.OUT/'study/runs'/pool).glob('*.json'));self.assertEqual(len(rows),n);self.assertTrue(all(s.read(x).get('status')=='PASS' for x in rows))
if __name__=='__main__':unittest.main()
