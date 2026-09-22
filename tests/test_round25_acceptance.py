import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round25_state as s
class Acceptance(unittest.TestCase):
 def test_dev_stops(self):
  d=s.read(s.OUT/'study/development-decision.json');self.assertFalse(d['eligible_b2r']);self.assertEqual(d['vetoes'],0)
  self.assertLess(d['b2r_without_best_two'],0)
 def test_downstream_skipped(self):
  p=s.read(s.OUT/'progress.json');self.assertEqual(p['steps']['R4']['status'],'SKIPPED_GATE');self.assertEqual(p['steps']['R5']['status'],'SKIPPED_GATE')
 def test_score_first(self):self.assertEqual(s.read(s.OUT/'progress.json')['objective'],'score_first')
if __name__=='__main__':unittest.main()
