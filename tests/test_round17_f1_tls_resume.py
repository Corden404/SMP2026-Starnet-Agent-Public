from pathlib import Path
import ssl, sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round17_f1_tls_resume as r

class F1TLSResumeTests(unittest.TestCase):
    def test_tls_adapter_verification_is_not_disabled(self):
        a=r.tls.TLSCompatAdapter()
        self.assertEqual(a.ssl_context.verify_mode,ssl.CERT_REQUIRED)
        self.assertTrue(a.ssl_context.check_hostname)
    def test_fingerprint_binds_transport_and_recovery_sources(self):
        fp=r.fingerprint();self.assertIn('base',fp);self.assertIn('new',fp)
        self.assertIn('workspace/round17_tls.py',fp['new']['files'])
    def test_existing_failed_history_is_transport_uncertain(self):
        p=r.base.OUT/'attempts/r13-glm-036-F1-attempt2.http.jsonl'
        self.assertTrue(r.base._latest_transport_uncertain(p))

if __name__=='__main__':unittest.main()
