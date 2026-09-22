import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round23_state as s
class Acceptance(unittest.TestCase):
 def test_r1_stopped(self):
  x=s.read(s.OUT/'engineering/summary.json')
  self.assertIsNone(x['selected']);self.assertEqual(x['eligible'],[])
  for v in ('E1','E2'):
   self.assertTrue(x['exactness'][v]['order_exact']);self.assertEqual(x['exactness'][v]['max_gain_error'],0.0)
 def test_frozen_gates(self):
  p=s.read(s.OUT/'engineering/protocol.json')
  self.assertEqual(p['performance_gate'],{'median_vs_orig':1.6,'min_cycle_vs_orig':1.5,'median_vs_sf1':1.08})
 def test_progress_skips_downstream(self):
  p=s.read(s.OUT/'progress.json')
  self.assertEqual(p['steps']['R1']['status'],'STOPPED_GATE')
  for k in ('R2','R3','R4','R5'):self.assertEqual(p['steps'][k]['status'],'SKIPPED_GATE')
 def test_real_model_order(self):
  self.assertEqual(s.REAL_MODEL_ORDER,('glm-4.6v','glm-4.5-air'))
if __name__=='__main__':unittest.main()
