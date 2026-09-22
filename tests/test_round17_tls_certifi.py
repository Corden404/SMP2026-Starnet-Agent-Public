from pathlib import Path
import ssl,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round17_tls_certifi as t

class CertifiTLSTests(unittest.TestCase):
 def test_verified_context(self):
  a=t.CertifiTLSAdapter()
  self.assertEqual(a.ssl_context.verify_mode,ssl.CERT_REQUIRED)
  self.assertTrue(a.ssl_context.check_hostname)
  self.assertGreaterEqual(a.ssl_context.minimum_version,ssl.TLSVersion.TLSv1_2)
 def test_persistent_session_adapter(self):
  s=t.session()
  try:self.assertIsInstance(s.get_adapter('https://open.bigmodel.cn'),t.CertifiTLSAdapter)
  finally:s.close()
if __name__=='__main__':unittest.main()
