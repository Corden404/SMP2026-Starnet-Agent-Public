from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round17_f1_tls_resume_v2 as r

class F1TLSV2Tests(unittest.TestCase):
 def test_existing_two_attempts_parse(self):
  entry=r.base.check()['seeds'][36]
  rows=r.tls1.history_for(entry,'F1')
  completed,pending,extra=r.replay.compose_history(rows,r.MODEL)
  self.assertEqual(len(completed),17);self.assertIsNotNone(pending);self.assertEqual(extra,2)
 def test_existing_latest_is_transport_uncertain(self):
  entry=r.base.check()['seeds'][36];rows=r.tls1.history_for(entry,'F1')
  self.assertTrue(r.replay.latest_transport_uncertain(rows[-1][1],r.base.TRANSPORT_ERRORS))
 def test_no_score_field_in_prepare_contract(self):
  self.assertNotIn('score',r.prepare.__doc__ or '')

if __name__=='__main__':unittest.main()
