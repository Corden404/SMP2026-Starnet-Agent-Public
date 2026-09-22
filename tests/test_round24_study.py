import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round24_study as st
class Study(unittest.TestCase):
 def test_counts(self):
  self.assertEqual(len(st.specs('dev')),54);self.assertEqual(len(st.specs('integration')),6);self.assertEqual(len(st.specs('confirm')),216);self.assertEqual(len(st.specs('stress')),54)
 def test_dev_reps(self):self.assertTrue(all(x['rep']==240 for x in st.specs('dev')))
 def test_gate_score_only(self):
  raw=dict(mean=1,without_best_two=.1,wins=4,ties=49,losses=1);x=dict(attributable=raw,prefix_drift=0,max_operations=80,max_calls=40,max_forward_error=0)
  self.assertTrue(st.eligible_dev(x))
 def test_gate_rejects_loss_balance(self):
  raw=dict(mean=1,without_best_two=.1,wins=1,ties=49,losses=4);x=dict(attributable=raw,prefix_drift=0,max_operations=80,max_calls=40,max_forward_error=0)
  self.assertFalse(st.eligible_dev(x))
if __name__=='__main__':unittest.main()
