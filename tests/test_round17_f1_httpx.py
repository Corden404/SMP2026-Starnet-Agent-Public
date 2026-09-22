from pathlib import Path
import ssl,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round17_f1_httpx as r
class F1HTTPXTests(unittest.TestCase):
 def test_prior_37_history_parses(self):
  e=r.base.check()['seeds'][37];rows=r.prev.history_for(e,'F1')
  completed,pending,extra=r.replay.compose_history(rows,r.MODEL)
  self.assertEqual(len(completed),0);self.assertIsNotNone(pending);self.assertEqual(extra,3)
 def test_httpx_context_verified(self):
  c=r.transport.context();self.assertEqual(c.verify_mode,ssl.CERT_REQUIRED);self.assertTrue(c.check_hostname)
if __name__=='__main__':unittest.main()
