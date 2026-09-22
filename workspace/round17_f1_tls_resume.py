"""TLS-compatible continuation of the still-frozen F1 real48 validation.

This is an operational recovery amendment only. It was registered after the
cause of repeated failures was identified as OpenSSL rejecting a weak-key
certificate, and after a CA/hostname-verifying SECLEVEL=1 canary succeeded.
No partial F1 score is read or used to change strategy/model/seeds.
"""
from __future__ import annotations
import argparse, contextlib, copy, io, json, time, types
from pathlib import Path
import requests

import round16_common as c
import round17_f1_complete as base
import round17_tls as tls
from round11_api import ServiceAbort, config
from round9_api import append, key
from round9_lock import ExclusiveBatch
from zhipu import ZhipuLLM, ensure_credit_available
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv

OUT=c.ROOT/'agent_notes/experiments/round17-f1-tls-resume'
MODEL=base.MODEL
MAX_ADDITIONAL_ATTEMPTS=3
DEPENDENCIES=('workspace/round17_f1_tls_resume.py','tests/test_round17_f1_tls_resume.py',
              'workspace/round17_tls.py','workspace/round17_f1_complete.py')


def fingerprint():
    return dict(base=base.fingerprint(),new=c.fingerprint(DEPENDENCIES))


def prepare():
    base.check();c.preserved()
    canary=c.ROOT/'agent_notes/experiments/round17-service-canary-tlscompat/glm-4.6v.http.jsonl'
    if not canary.exists():raise base.RecoveryMismatch('TLS compatibility canary missing')
    events=base._events(canary)
    if not any(e.get('event')=='response' and e.get('status')==200 for e in events):
        raise base.RecoveryMismatch('TLS compatibility canary did not pass')
    prior=[base.OUT/'protocol.json',base.OUT/'BLOCKED.json',
           base.OUT/'attempts/r13-glm-036-F1-attempt1.json',
           base.OUT/'attempts/r13-glm-036-F1-attempt1.http.jsonl',
           base.OUT/'attempts/r13-glm-036-F1-attempt2.json',
           base.OUT/'attempts/r13-glm-036-F1-attempt2.http.jsonl']
    protocol=dict(code=fingerprint(),model=MODEL,model_config=config(MODEL),seeds=base.check()['seeds'],
        planned_graphs=48,additional_attempt_cap=MAX_ADDITIONAL_ATTEMPTS,
        cause='Transport-only amendment after verified CERTIFICATE_VERIFY_FAILED: EE certificate key too weak.',
        tls='OpenSSL SECLEVEL=1 with CERT_REQUIRED, hostname checking and TLS>=1.2; verify=False is forbidden.',
        canary_sha256=c.sha(canary),protected_prior={x.relative_to(c.ROOT).as_posix():c.sha(x) for x in prior},
        score_blind=True,no_model_change=True,no_seed_change=True,no_policy_change=True,no_upload=True)
    c.save(OUT/'protocol.json',protocol,True)
    print(json.dumps(dict(prepared=True,cause=protocol['cause'],tls=protocol['tls'],originals=c.preserved()),ensure_ascii=False))


def check():
    p=c.read(OUT/'protocol.json')
    if p['code']!=fingerprint():raise base.RecoveryMismatch('TLS-resume source drift')
    canary=c.ROOT/'agent_notes/experiments/round17-service-canary-tlscompat/glm-4.6v.http.jsonl'
    if c.sha(canary)!=p['canary_sha256']:raise base.RecoveryMismatch('Canary evidence changed')
    for name,h in p['protected_prior'].items():
        if c.sha(c.ROOT/name)!=h:raise base.RecoveryMismatch('Prior evidence changed: '+name)
    return p


def history_for(entry,variant):
    records=base.history_for(entry,variant)
    name=entry['spec']['case']+'-'+variant
    for i in range(1,MAX_ADDITIONAL_ATTEMPTS+1):
        rp=OUT/'attempts'/(name+f'-attempt{i}.json')
        if not rp.exists():break
        row=c.read(rp)
        if row.get('status')!='FAILED':raise base.RecoveryMismatch('Unexpected TLS attempt state')
        records.append((row,rp.with_suffix('.http.jsonl')))
    return records


class TLSReplayLLM(base.CumulativeReplayLLM):
    def _live_post(self,url,data):
        if self.http_request_count>=119:raise ServiceAbort('Physical API limit')
        self.http_request_count+=1;number=self.http_request_count
        append(self.path,dict(event='request',number=number,data=data))
        try:
            response=tls.post(url,headers=self.headers,json=data,timeout=(10,45),allow_redirects=False)
        except requests.RequestException as exc:
            append(self.path,dict(event='uncertain',number=number,error_type=type(exc).__name__,may_be_billed=True))
            raise ServiceAbort('Uncertain API result; no automatic retry') from exc
        append(self.path,dict(event='response',number=number,status=response.status_code,body=response.text))
        ensure_credit_available(response)
        if response.status_code!=200:raise ServiceAbort('HTTP_'+str(response.status_code))
        return response

    def _post_once(self,url,data):
        if self.cursor<len(self.completed):
            request,response=self.completed[self.cursor]
            if request['data']!=data:raise base.RecoveryMismatch('Historical successful request input changed')
            self.cursor+=1;self.http_request_count+=1
            return types.SimpleNamespace(status_code=200,text=response['body'],json=lambda:json.loads(response['body']))
        if not self.boundary_checked:
            if self.pending is not None and self.pending['data']!=data:
                raise base.RecoveryMismatch('Latest uncertain request input changed')
            if self.boundary is not None:self.boundary()
            self.boundary_checked=True
            for _ in range(self.extra_uncertain):
                self.http_request_count+=1
                if self.counter is not None:self.counter._reserve_call()
            if not self.live:raise base.parent.ReplayComplete()
        if not self.live:raise base.parent.ReplayComplete()
        self.new_requests+=1
        return self._live_post(url,data)


def run_attempt(entry,variant,attempt):
    spec=entry['spec'];name=spec['case']+'-'+variant;records=history_for(entry,variant)
    completed,pending,extra=base.compose_history(records) if records else ([],None,0)
    latest=records[-1][0] if records else None
    target=OUT/'attempts'/(name+f'-attempt{attempt}.json');journal=target.with_suffix('.http.jsonl')
    if target.exists() or journal.exists():raise base.RecoveryMismatch('TLS attempt already exists')
    module=base.parent.load(variant);module.coefficients.cache_clear()
    raw=FiniteEnv(c.generate(spec));logged=LoggedEnv(raw);chains=[]
    llm=TLSReplayLLM(journal,completed,pending,extra,True)
    row=dict(status='STARTED',identity=dict(entry=entry,variant=variant,model_config=config(MODEL),code=check()['code']),
             spec=spec,variant=variant,attempt=attempt)
    c.save(target,row);started=time.monotonic()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model=c.old.initialize(module,logged,mode='integration',llm=llm);llm.counter=model.llm_counter
            def boundary():
                if latest is None:return
                if not base.same_prefix(logged.actions,latest['actions'],chains,latest['chains']):
                    raise base.RecoveryMismatch('Public prefix/chains changed before TLS retry')
            llm.boundary=boundary
            for role in ('scout','commander'):
                agent=getattr(model,role);orig=agent._run_chain
                def trace(name_,data,orig=orig,role=role):
                    before=copy.deepcopy(data);ans=orig(name_,data)
                    chains.append(dict(role=role,chain=name_,operation=model.operation_count,input=before,output=copy.deepcopy(ans)))
                    return ans
                agent._run_chain=trace
            result=c.finish(raw,logged,model)
        if records and not llm.boundary_checked:raise base.RecoveryMismatch('TLS recovery boundary not checked')
        if len(raw._scanned)!=44 or llm.http_request_count>model.llm_counter.call_count or model.llm_counter.call_count>=120:
            raise base.RecoveryMismatch('TLS recovery resource accounting mismatch')
        row.update(result,status='PASS',chains=chains,http_calls=llm.http_request_count,new_http_calls=llm.new_requests,
            replayed_responses=llm.cursor,charged_uncertain_attempts=extra,metadata=llm.response_metadata,
            elapsed_seconds=time.monotonic()-started)
        c.save(target,row);c.save(OUT/'final'/(name+'.json'),row,True);return row
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,actions=logged.actions,chains=chains,
            new_http_calls=llm.new_requests,replayed_responses=llm.cursor,charged_uncertain_attempts=extra,
            elapsed_seconds=time.monotonic()-started)
        c.save(target,row);raise


def existing_pass(entry,variant):
    found=base.existing_pass(entry,variant)
    name=entry['spec']['case']+'-'+variant+'.json';path=OUT/'final'/name
    if path.exists():
        row=c.read(path)
        if row.get('status')!='PASS' or row['spec']!=entry['spec'] or row['variant']!=variant or row['identity']['model_config']!=config(MODEL):
            raise base.RecoveryMismatch('TLS final identity mismatch')
        c.audit_game(row)
        if found is not None:raise base.RecoveryMismatch('Duplicate PASS across recovery layers')
        found=(path,row)
    return found


def complete_one(index,variant):
    p=check();entry=p['seeds'][index]
    found=existing_pass(entry,variant)
    if found:return found[1]
    records=history_for(entry,variant)
    if records and not base._latest_transport_uncertain(records[-1][1]):
        raise ServiceAbort('Prior failure is not transport-recoverable')
    existing_tls=sum(1 for rp,jp in records if str(rp.get('identity',{}).get('code'))==str(p['code']))
    start=existing_tls+1
    for attempt in range(start,MAX_ADDITIONAL_ATTEMPTS+1):
        try:return run_attempt(entry,variant,attempt)
        except ServiceAbort:
            rp=OUT/'attempts'/(entry['spec']['case']+'-'+variant+f'-attempt{attempt}.json');jp=rp.with_suffix('.http.jsonl')
            if not jp.exists() or not base._latest_transport_uncertain(jp) or attempt>=MAX_ADDITIONAL_ATTEMPTS:
                c.save(OUT/'BLOCKED.json',dict(case=entry['spec']['case'],variant=variant,attempt=attempt,reason='service_or_tls_cap'))
                raise
    raise ServiceAbort('TLS recovery cap exhausted')


def run(start=36,stop=48):
    check()
    with ExclusiveBatch(OUT/'execution.lock'):
        for index in range(start,stop):
            for variant in (('S44','F1') if index%2==0 else ('F1','S44')):
                row=complete_one(index,variant)
                print(json.dumps(dict(index=index,variant=variant,status=row['status'],new_http_calls=row.get('new_http_calls',0),
                    charged_uncertain=row.get('charged_uncertain_attempts',0))),flush=True)
    return summarize(False)


def summarize(require_all=True):
    p=check();pairs=[];missing=[];completed=0;max_error=0.
    for entry in p['seeds']:
        group={}
        for variant in ('S44','F1'):
            found=existing_pass(entry,variant)
            if found is None:missing.append(entry['spec']['case']+'-'+variant);continue
            row=found[1];group[variant]=row;completed+=1;max_error=max(max_error,c.audit_game(row))
        if len(group)==2:pairs.append(dict(case=entry['spec']['case'],family=entry['spec']['family'],delta=group['F1']['final_score']-group['S44']['final_score']))
    result=dict(status='COMPLETE' if not missing else 'INCOMPLETE',planned_graphs=48,completed_games=completed,
        complete_pairs=len(pairs),missing=missing,max_forward_error=max_error,model=MODEL,tls_compat_verified=True,originals=c.preserved())
    if not missing:result.update(statistics=c.paired_stats(pairs),pairs=pairs)
    c.save(OUT/'summary.json',result)
    if require_all and missing:raise base.RecoveryMismatch('F1 TLS cohort incomplete')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','missing')},ensure_ascii=False),flush=True);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary'])
    ap.add_argument('--start',type=int,default=36);ap.add_argument('--stop',type=int,default=48);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run(a.start,a.stop)
    else:summarize(True)

if __name__=='__main__':main()
