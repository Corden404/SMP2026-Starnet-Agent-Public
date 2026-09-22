import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round21_study as st
class Study(unittest.TestCase):
 def test_fresh_protocol_counts(self):
  seeds={p:st.h.entries(p) for p in st.POOLS};self.assertEqual(len(seeds['dev']),54);self.assertEqual(len(seeds['confirm']),216);self.assertEqual(len(seeds['stress']),54);self.assertEqual(len(seeds['integration']),6)
 def test_safe_rejects_unexplained(self):
  x=dict(prefix_drift=0,nontriggered_different=0,unexplained_changed=1,max_operations=80,max_calls=40,max_forward_error=0);self.assertFalse(st.safe(x))
 def test_safe_accepts_clean(self):
  x=dict(prefix_drift=0,nontriggered_different=0,unexplained_changed=0,max_operations=80,max_calls=40,max_forward_error=0);self.assertTrue(st.safe(x))
if __name__=='__main__':unittest.main()
