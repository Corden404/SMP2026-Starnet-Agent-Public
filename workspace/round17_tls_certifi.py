"""Stable verified TLS transport for validation harnesses.

Uses certifi's explicit CA bundle, TLS>=1.2, hostname checking, CERT_REQUIRED,
and OpenSSL SECLEVEL=1. A session is intentionally persistent per model run.
"""
import ssl, certifi, requests
from requests.adapters import HTTPAdapter

class CertifiTLSAdapter(HTTPAdapter):
    def __init__(self,*args,**kwargs):
        self.ssl_context=ssl.create_default_context(cafile=certifi.where())
        self.ssl_context.minimum_version=ssl.TLSVersion.TLSv1_2
        self.ssl_context.check_hostname=True
        self.ssl_context.verify_mode=ssl.CERT_REQUIRED
        self.ssl_context.set_ciphers('DEFAULT@SECLEVEL=1')
        super().__init__(*args,**kwargs)
    def init_poolmanager(self,connections,maxsize,block=False,**kw):
        kw['ssl_context']=self.ssl_context
        return super().init_poolmanager(connections,maxsize,block=block,**kw)
    def proxy_manager_for(self,proxy,**kw):
        kw['ssl_context']=self.ssl_context
        return super().proxy_manager_for(proxy,**kw)

def session():
    s=requests.Session()
    s.mount('https://',CertifiTLSAdapter())
    return s
