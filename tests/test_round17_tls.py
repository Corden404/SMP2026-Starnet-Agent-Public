from pathlib import Path
import ssl, sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round17_tls as t

class TLSCompatTests(unittest.TestCase):
    def test_verification_remains_enabled(self):
        a=t.TLSCompatAdapter()
        self.assertEqual(a.ssl_context.verify_mode,ssl.CERT_REQUIRED)
        self.assertTrue(a.ssl_context.check_hostname)
        self.assertGreaterEqual(a.ssl_context.minimum_version,ssl.TLSVersion.TLSv1_2)
    def test_session_mounts_custom_adapter(self):
        s=t.session()
        try:self.assertIsInstance(s.get_adapter('https://example.com'),t.TLSCompatAdapter)
        finally:s.close()

if __name__=='__main__':unittest.main()
