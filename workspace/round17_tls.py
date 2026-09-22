"""TLS compatibility transport for validation harnesses only.

Lowers OpenSSL cipher security level to 1 to accept the observed weak-key
certificate while retaining CA validation and hostname checking. Never disables
certificate verification and is not part of participant submission code.
"""
import ssl
import requests
from requests.adapters import HTTPAdapter

class TLSCompatAdapter(HTTPAdapter):
    def __init__(self,*args,**kwargs):
        self.ssl_context=ssl.create_default_context()
        self.ssl_context.minimum_version=ssl.TLSVersion.TLSv1_2
        self.ssl_context.check_hostname=True
        self.ssl_context.verify_mode=ssl.CERT_REQUIRED
        self.ssl_context.set_ciphers('DEFAULT@SECLEVEL=1')
        super().__init__(*args,**kwargs)
    def init_poolmanager(self,connections,maxsize,block=False,**pool_kwargs):
        pool_kwargs['ssl_context']=self.ssl_context
        return super().init_poolmanager(connections,maxsize,block=block,**pool_kwargs)
    def proxy_manager_for(self,proxy,**proxy_kwargs):
        proxy_kwargs['ssl_context']=self.ssl_context
        return super().proxy_manager_for(proxy,**proxy_kwargs)

def session():
    s=requests.Session()
    s.mount('https://',TLSCompatAdapter())
    return s

def post(url,**kwargs):
    with session() as s:
        return s.post(url,verify=True,**kwargs)
