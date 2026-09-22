import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round24_state as s
class Acceptance(unittest.TestCase):
 def test_score_first(self):
  p=s.read(s.OUT/'progress.json');self.assertEqual(p['objective'],'score_first')
 def test_dev_gate_stopped(self):
  d=s.read(s.OUT/'study/development-decision.json');self.assertIsNone(d['selected']);self.assertEqual(d['eligible'],[])
  for v in ('B2B','B2M2','B2M3'):
   self.assertGreater(d['candidates'][v]['raw_mean'],0);self.assertLess(d['candidates'][v]['without_best_two'],0)
 def test_real_skipped(self):
  p=s.read(s.OUT/'progress.json');self.assertEqual(p['steps']['R4']['status'],'SKIPPED_GATE');self.assertEqual(p['steps']['R5']['status'],'SKIPPED_GATE')
if __name__=='__main__':unittest.main()
