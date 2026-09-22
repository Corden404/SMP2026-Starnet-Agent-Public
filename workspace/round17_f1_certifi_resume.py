"""Final F1 real48 service-resume layer using certifi + persistent verified TLS."""
from __future__ import annotations
import argparse, contextlib, copy, io, json, time, types
from pathlib import Path
import requests

import round16_common as c
import round17_f1_complete as base
import round17_f1_tls_resume_v2 as prev
import round17_replay as replay
import round17_tls_certifi as tls
from round11_api import ServiceAbort, config
from round9_api import append
from round9_lock import ExclusiveBatch
from zhipu import ensure_credit_available
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv

OUT=c.ROOT/'agent_notes/experiments/round17-f1-certifi-resume'
MODEL=base.MODEL
MAX_ADDITIONAL_ATTEMPTS=3
DEPENDENCIES=('workspace/round17_f1_certifi_resume.py','tests/test_round17_f1_certifi_resume.py',
              'workspace/round17_tls_certifi.py','workspace/round17_replay.py',
              'workspace/round17_f1_tls_resume_v2.py')


def fingerprint():return dict(prev=prev.fingerprint(),new=c.fingerprint(DEPENDENCIES))


def prepare():
    prev.check();c.preserved()
    entry=base.check()['seeds'][36]
    rows=prev.history_for(entry,'F1')
    completed,pending,extra=replay.compose_history(rows,MODEL)
    if (len(completed),extra,pending and pending['number'])!=(23,5,28):
        raise base.RecoveryMismatch('Unexpected pre-certifi recovery history')
    canary=c.ROOT/'agent_notes/experiments/round17-service-canary-certifi/glm-4.6v.http.jsonl'
    ev=replay.events(canary)
    if sum(e.get('event')=='response' and e.get('status')==200 for e in ev)!=3:
        raise base.RecoveryMismatch('Persistent certifi canary incomplete')
    protected=[prev.OUT/'protocol.json',prev.OUT/'BLOCKED.json',canary]
    protected += sorted((prev.OUT/'attempts').glob('r13-glm-036-F1-attempt*'))
    protocol=dict(code=fingerprint(),model=MODEL,model_config=config(MODEL),seeds=base.check()['seeds'],
        planned_graphs=48,additional_attempt_cap=MAX_ADDITIONAL_ATTEMPTS,
        reason='Service-only revision after 8/8 certifi TLS handshakes and 3/3 persistent-session API canary success.',
        history_check=dict(successful_responses=len(completed),uncertain_attempts=extra,pending_number=pending['number']),
        tls='certifi CA bundle; CERT_REQUIRED; hostname check; TLS>=1.2; SECLEVEL=1; one persistent Session per game',
        protected_prior={x.relative_to(c.ROOT).as_posix():c.sha(x) for x in protected},
        score_blind=True,no_model_change=True,no_seed_change=True,no_policy_change=True,no_upload=True)
    c.save(OUT/'protocol.json',protocol,True)
    print(json.dumps(dict(prepared=True,history_check=protocol['history_check'],tls=protocol['tls'],originals=c.preserved()),ensure_ascii=False))


def check():
    p=c.read(OUT/'protocol.json')
    if p['code']!=fingerprint():raise base.RecoveryMismatch('Certifi F1 source drift')
    for name,h in p['protected_prior'].items():
        if c.sha(c.ROOT/name)!=h:raise base.RecoveryMismatch('Protected prior changed: '+name)
    return p


def history_for(entry,variant):
    records=prev.history_for(entry,variant);name=entry['spec']['case']+'-'+variant
    for i in range(1,MAX_ADDITIONAL_ATTEMPTS+1):
        rp=OUT/'attempts'/(name+f'-attempt{i}.json')
        if not rp.exists():break
        row=c.read(rp)
        if row.get('status')!='FAILED':raise base.RecoveryMismatch('Unexpected certifi attempt state')
        records.append((row,rp.with_suffix('.http.jsonl')))
    return records


def existing_pass(entry,variant):
    found=prev.existing_pass(entry,variant)
    path=OUT/'final'/(entry['spec']['case']+'-'+variant+'.json')
    if path.exists():
        row=c.read(path)
        if row.get('status')!='PASS' or row['spec']!=entry['spec'] or row['variant']!=variant or row['identity']['model_config']!=config(MODEL):
            raise base.RecoveryMismatch('Certifi F1 final identity mismatch')
        c.audit_game(row)
        if found is not None:raise base.RecoveryMismatch('Duplicate PASS across recovery layers')
        found=(path,row)
    return found


class PersistentReplayLLM(prev.tls1.TLSReplayLLM):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._persistent=tls.session()
    def close(self):self._persistent.close()
    def _live_post(self,url,data):
        if self.http_request_count>=119:raise ServiceAbort('Physical API limit')
        self.http_request_count+=1;number=self.http_request_count
        append(self.path,dict(event='request',number=number,data=data))
        try:
            response=self._persistent.post(url,headers=self.headers,json=data,timeout=(10,45),allow_redirects=False,verify=True)
        except requests.RequestException as exc:
            append(self.path,dict(event='uncertain',number=number,error_type=type(exc).__name__,may_be_billed=True))
            raise ServiceAbort('Uncertain API result; no automatic retry') from exc
        append(self.path,dict(event='response',number=number,status=response.status_code,body=response.text))
        ensure_credit_available(response)
        if response.status_code!=200:raise ServiceAbort('HTTP_'+str(response.status_code))
        return response


def run_attempt(entry,variant,attempt):
    spec=entry['spec'];name=spec['case']+'-'+variant;records=history_for(entry,variant)
    completed,pending,extra=replay.compose_history(records,MODEL) if records else ([],None,0)
    latest=records[-1][0] if records else None
    target=OUT/'attempts'/(name+f'-attempt{attempt}.json');journal=target.with_suffix('.http.jsonl')
    if target.exists() or journal.exists():raise base.RecoveryMismatch('Certifi attempt already exists')
    module=base.parent.load(variant);module.coefficients.cache_clear()
    raw=FiniteEnv(c.generate(spec));logged=LoggedEnv(raw);chains=[]
    llm=PersistentReplayLLM(journal,completed,pending,extra,True)
    row=dict(status='STARTED',identity=dict(entry=entry,variant=variant,model_config=config(MODEL),code=check()['code']),
             spec=spec,variant=variant,attempt=attempt)
    c.save(target,row);started=time.monotonic()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model=c.old.initialize(module,logged,mode='integration',llm=llm);llm.counter=model.llm_counter
            def boundary():
                if latest is None:return
                if not base.same_prefix(logged.actions,latest['actions'],chains,latest['chains']):
                    raise base.RecoveryMismatch('Certifi F1 public prefix/chains changed')
            llm.boundary=boundary
            for role in ('scout','commander'):
                agent=getattr(model,role);orig=agent._run_chain
                def trace(name_,data,orig=orig,role=role):
                    before=copy.deepcopy(data);ans=orig(name_,data)
                    chains.append(dict(role=role,chain=name_,operation=model.operation_count,input=before,output=copy.deepcopy(ans)));return ans
                agent._run_chain=trace
            result=c.finish(raw,logged,model)
        if records and not llm.boundary_checked:raise base.RecoveryMismatch('Certifi F1 boundary not checked')
        if len(raw._scanned)!=44 or llm.http_request_count>model.llm_counter.call_count or model.llm_counter.call_count>=120:
            raise base.RecoveryMismatch('Certifi F1 resource accounting mismatch')
        row.update(result,status='PASS',chains=chains,http_calls=llm.http_request_count,new_http_calls=llm.new_requests,
            replayed_responses=llm.cursor,charged_uncertain_attempts=extra,metadata=llm.response_metadata,elapsed_seconds=time.monotonic()-started)
        c.save(target,row);c.save(OUT/'final'/(name+'.json'),row,True);return row
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,actions=logged.actions,chains=chains,
            new_http_calls=llm.new_requests,replayed_responses=llm.cursor,charged_uncertain_attempts=extra,elapsed_seconds=time.monotonic()-started)
        c.save(target,row);raise
    finally:llm.close()


def complete_one(index,variant):
    p=check();entry=p['seeds'][index];found=existing_pass(entry,variant)
    if found:return found[1]
    records=history_for(entry,variant)
    if records and not replay.latest_transport_uncertain(records[-1][1],base.TRANSPORT_ERRORS):
        raise ServiceAbort('Prior failure is not transport-recoverable')
    own=sum((OUT/'attempts'/(entry['spec']['case']+'-'+variant+f'-attempt{i}.json')).exists() for i in range(1,MAX_ADDITIONAL_ATTEMPTS+1))
    for attempt in range(own+1,MAX_ADDITIONAL_ATTEMPTS+1):
        try:return run_attempt(entry,variant,attempt)
        except ServiceAbort:
            rp=OUT/'attempts'/(entry['spec']['case']+'-'+variant+f'-attempt{attempt}.json');jp=rp.with_suffix('.http.jsonl')
            if not jp.exists() or not replay.latest_transport_uncertain(jp,base.TRANSPORT_ERRORS) or attempt>=MAX_ADDITIONAL_ATTEMPTS:
                c.save(OUT/'BLOCKED.json',dict(case=entry['spec']['case'],variant=variant,attempt=attempt,reason='service_or_certifi_cap'));raise
    raise ServiceAbort('Certifi F1 recovery cap exhausted')


def run(start=36,stop=48):
    check()
    if not 0<=start<stop<=48:raise ValueError('Invalid range')
    with ExclusiveBatch(OUT/'execution.lock'):
        for index in range(start,stop):
            for variant in (('S44','F1') if index%2==0 else ('F1','S44')):
                row=complete_one(index,variant)
                print(json.dumps(dict(index=index,variant=variant,status=row['status'],new_http_calls=row.get('new_http_calls',0),charged_uncertain=row.get('charged_uncertain_attempts',0))),flush=True)
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
    result=dict(status='COMPLETE' if not missing else 'INCOMPLETE',planned_graphs=48,completed_games=completed,complete_pairs=len(pairs),
        missing=missing,max_forward_error=max_error,model=MODEL,tls='certifi-persistent',originals=c.preserved())
    if not missing:result.update(statistics=c.paired_stats(pairs),pairs=pairs)
    c.save(OUT/'summary.json',result)
    if require_all and missing:raise base.RecoveryMismatch('Certifi F1 cohort incomplete')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','missing')},ensure_ascii=False),flush=True);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary']);ap.add_argument('--start',type=int,default=36);ap.add_argument('--stop',type=int,default=48);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run(a.start,a.stop)
    else:summarize(True)
if __name__=='__main__':main()
