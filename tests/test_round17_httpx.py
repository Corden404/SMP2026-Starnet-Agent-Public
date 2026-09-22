from pathlib import Path
import ssl,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round17_httpx as t
class HTTPXTransportTests(unittest.TestCase):
 def test_context_is_verified(self):
  c=t.context();self.assertEqual(c.verify_mode,ssl.CERT_REQUIRED);self.assertTrue(c.check_hostname);self.assertGreaterEqual(c.minimum_version,ssl.TLSVersion.TLSv1_2)
 def test_client_has_no_environment_proxy_inheritance(self):
  c=t.client()
  try:self.assertFalse(getattr(c,'_trust_env',True))
  finally:c.close()
if __name__=='__main__':unittest.main()
