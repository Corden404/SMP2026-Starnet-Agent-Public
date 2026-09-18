"""Sandbox journal, circuit breaker and free health recovery. POSTs are never retried in-session."""
import json
import math
import random
import time
import requests

from api_client import RemoteStarNetEnv
from recovery_store import RecoveryStop,InfrastructureFailure,encode


def semantic(value):
    if isinstance(value,dict):
        return {k:sorted(v) if k=='neighbors' else semantic(v) for k,v in value.items()
                if k not in ('session_id','request_id','timestamp')}
    if isinstance(value,list):return [semantic(x) for x in value]
    return value


class Gate:
    def __init__(self,store,clock=time.time,sleep=time.sleep):
        self.store=store;self.clock=clock;self.sleep=sleep;self.last_request=0

    def check(self,paid=False,health=False):
        state=self.store.get('circuit',{'phase':'STARTING'})
        if state.get('deadline') and self.clock()>=state['deadline']:
            self.stop('THREE_HOUR_OUTAGE')
        if state['phase']=='BLOCKED':raise RecoveryStop(state.get('reason','BLOCKED'))
        if not health and state['phase'] not in ('READY','RECOVERING'):
            raise RecoveryStop('PAID_GATE_CLOSED' if paid else 'SANDBOX_GATE_CLOSED')

    def pace(self,health=False):
        self.check(health=health)
        delay=max(0,1-(self.clock()-self.last_request))
        if delay:self.sleep(delay)
        self.check(health=health);self.last_request=self.clock()

    def stop(self,reason):
        state=self.store.get('circuit',{})
        state.update(phase='BLOCKED',reason=reason,stopped_at=self.clock());self.store.put('circuit',state)
        raise RecoveryStop(reason)

    def outage(self,detail,trial_failure=False,retry_after=0):
        now=self.clock();state=self.store.get('circuit',{})
        if not state.get('deadline'):
            state.update(recovery_started_at=now,deadline=now+10800,failed_recoveries=0)
        if trial_failure and state.get('phase')=='RECOVERING':
            state['failed_recoveries']=state.get('failed_recoveries',0)+1
        state.update(phase='WAITING',last_failure_at=now,last_error=detail,
                     retry_after_until=max(state.get('retry_after_until',0),now+retry_after))
        self.store.put('circuit',state)
        if state.get('failed_recoveries',0)>=3:self.stop('PERSISTENT_SANDBOX_FAILURE')

    def completed(self):
        old=self.store.get('circuit',{})
        self.store.put('last_recovered_window',old)
        self.store.put('circuit',{'phase':'READY','last_completed_at':self.clock()})

    def wait(self,seconds):
        until=self.clock()+seconds
        while self.clock()<until:
            self.check(health=True)
            self.sleep(min(1,until-self.clock()))
        self.check(health=True)

    def recover(self,probe,canary):
        successes=0;tries=0
        while True:
            state=self.store.get('circuit',{'phase':'STARTING'})
            self.check(health=True)
            if tries or state['phase']=='WAITING':
                backoff=60 if successes else (60,120,300)[min(tries,2)]
                seconds=backoff*random.uniform(.8,1.2)
                seconds=max(seconds,state.get('retry_after_until',0)-self.clock())
                self.wait(seconds)
            try:
                probe();successes+=1;tries+=1
                self.store.put('health_progress',{'successes':successes,'checks':tries,'time':self.clock()})
                if successes<3:continue
                canary()
                state=self.store.get('circuit',{})
                state['phase']='RECOVERING' if state.get('deadline') else 'READY'
                self.store.put('circuit',state);return
            except InfrastructureFailure as e:
                successes=0;tries+=1;self.outage(e.detail)


class Transport:
    def __init__(self,store,gate,aid=0,health=False,post=None):
        self.store=store;self.gate=gate;self.aid=aid;self.health=health
        self.session=requests.Session();self.post=post or self.session.post

    def send(self,url,payload):
        self.gate.pace(health=self.health)
        wid=self.store.begin_wire(self.aid,'sandbox',{'url':url,'payload':payload})
        try:
            response=self.post(url,json=payload,timeout=(10,30))
            detail={'status':response.status_code,'text':response.text,
                    'headers':{k:v for k,v in response.headers.items() if k.lower() in
                               ('retry-after','server','via','x-request-id','content-type')}}
            self.store.end_wire(wid,'RESPONSE_DURABLE',detail)
        except requests.RequestException as e:
            self.store.end_wire(wid,'ENV_RESULT_UNKNOWN',{'error_type':type(e).__name__,'endpoint':url})
            if isinstance(e,requests.exceptions.SSLError):raise RecoveryStop('TLS_FAILURE',url)
            raise InfrastructureFailure(type(e).__name__+' '+url)
        if response.status_code in (500,502,503,504,408,429):
            retry=response.headers.get('Retry-After','0')
            try:retry=float(retry)
            except ValueError:
                from email.utils import parsedate_to_datetime
                try:retry=max(0,parsedate_to_datetime(retry).timestamp()-self.gate.clock())
                except (ValueError,TypeError):retry=0
            self.store.put('last_retry_after',retry)
            raise InfrastructureFailure('HTTP '+str(response.status_code)+' '+url)
        # A confirmed expired session must be discarded, including free probes.
        # Other 404 responses remain protocol failures; never retry in-session.
        if response.status_code==404:
            try:expired=response.json()=={'error':'Session expired'}
            except ValueError:expired=False
            if expired:raise InfrastructureFailure('Session expired '+url)
        if response.status_code in (401,403):raise RecoveryStop('SANDBOX_AUTH_FAILURE',url)
        if not 200<=response.status_code<300:raise RecoveryStop('SANDBOX_PROTOCOL_FAILURE',encode(detail))
        try:data=response.json()
        except ValueError:raise RecoveryStop('SANDBOX_PROTOCOL_FAILURE','Invalid JSON '+url)
        if not isinstance(data,dict):raise RecoveryStop('SANDBOX_PROTOCOL_FAILURE','Non-object '+url)
        return data


class DurableEnv(RemoteStarNetEnv):
    def __init__(self,url,seed,journal,transport):
        self.journal=journal;self.transport=transport;self.quarantined=False
        self.actions=[];self.last_budget=None
        super().__init__(url,seed,timeout=30)

    def _post(self,path,payload):
        if self.quarantined:raise RecoveryStop('SESSION_QUARANTINED')
        request={'path':path,'payload':{k:v for k,v in payload.items() if k!='session_id'}}
        old=None
        if path!='/api/start_session':
            old=self.journal.enter('sandbox',request);self.journal.inflight()
        try:
            data=self.transport.send(self.api_url+path,payload)
            clean=semantic(data)
            if old and old['response'] is not None and json.loads(old['response'])!=clean:
                self.journal.store.put('last_divergence',{'path':path,'old':json.loads(old['response']),'new':clean})
                raise RecoveryStop('REPLAY_DIVERGENCE',path+' response changed')
            if path!='/api/start_session':self.journal.response(clean)
            if path=='/api/get_budget':
                budget=data.get('budget')
                if type(budget) not in (int,float) or not math.isfinite(budget) or budget<0:
                    raise RecoveryStop('SANDBOX_PROTOCOL_FAILURE','Invalid budget')
                self.last_budget=budget
            return data
        except BaseException:
            self.quarantined=True;raise
