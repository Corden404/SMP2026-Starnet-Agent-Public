import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round22_state as s
class Acceptance(unittest.TestCase):
 def test_timing_and_causal(self):
  t=s.read(s.OUT/'timing/summary.json');c=s.read(s.OUT/'causal/summary.json')
  self.assertTrue(t['intrinsic_timing_drift']);self.assertTrue(t['stress031_pre20_drift'])
  self.assertTrue(c['eligible_r3']);self.assertGreater(c['mean_delta'],0);self.assertGreater(c['without_best_two'],0)
 def test_stablefast_stops_at_frozen_gate(self):
  x=s.read(s.OUT/'stablefast/summary.json');self.assertTrue(x['exactness']['order_exact']);self.assertEqual(x['exactness']['max_gain_error'],0)
  self.assertLess(x['performance']['speedup'],1.5);self.assertFalse(x['eligible_sf'])
 def test_progress_skips_r4(self):
  p=s.read(s.OUT/'progress.json');self.assertEqual(p['steps']['R3']['status'],'STOPPED_GATE');self.assertEqual(p['steps']['R4']['status'],'SKIPPED_GATE')
if __name__=='__main__':unittest.main()
