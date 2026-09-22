"""Verified httpx transport with an explicit SSLContext for real-model validation."""
import ssl, certifi, httpx

def context():
    ctx=ssl.create_default_context(cafile=certifi.where())
    ctx.minimum_version=ssl.TLSVersion.TLSv1_2
    ctx.check_hostname=True
    ctx.verify_mode=ssl.CERT_REQUIRED
    ctx.set_ciphers('DEFAULT@SECLEVEL=1')
    return ctx

def client():
    return httpx.Client(verify=context(),trust_env=False,
        timeout=httpx.Timeout(45.0,connect=10.0),follow_redirects=False)
