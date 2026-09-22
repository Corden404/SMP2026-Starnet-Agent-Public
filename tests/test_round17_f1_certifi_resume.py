from pathlib import Path
import ssl,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round17_f1_certifi_resume as r
class CertifiF1Tests(unittest.TestCase):
 def test_pre_history_shape(self):
  e=r.base.check()['seeds'][36];rows=r.prev.history_for(e,'F1')
  completed,pending,extra=r.replay.compose_history(rows,r.MODEL)
  self.assertEqual((len(completed),extra,pending['number']),(23,5,28))
 def test_persistent_transport_keeps_verification(self):
  llm=r.PersistentReplayLLM('unused',[],None,0,False)
  try:
   a=llm._persistent.get_adapter('https://open.bigmodel.cn')
   self.assertEqual(a.ssl_context.verify_mode,ssl.CERT_REQUIRED);self.assertTrue(a.ssl_context.check_hostname)
  finally:llm.close()
if __name__=='__main__':unittest.main()
