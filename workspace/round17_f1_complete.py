"""Complete the frozen F1 real-LLM cohort with bounded, evidence-based transport recovery.

Existing PASS games are reused. The round16 SSL failure is never overwritten.
A transport-uncertain request is charged to both physical/logical counters, then
the whole public prefix is replayed exactly before one replacement request.
HTTP/credit/payload/identity failures stop the cohort. No model/seed switching.
"""
from __future__ import annotations
import argparse, contextlib, copy, io, json, time, types
from pathlib import Path

import round16_common as c
import round16_f1_blocks as blocks
import round16_f1_recovery as parent
from round11_api import JournalLLM, ServiceAbort, config
from round9_api import key
from round9_lock import ExclusiveBatch
from zhipu import ZhipuLLM
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv

OUT=c.ROOT/'agent_notes/experiments/round17-f1-complete'
MODEL=blocks.MODEL
MAX_TRANSPORT_RECOVERIES=2
TRANSPORT_ERRORS={'SSLError','ConnectionError','ConnectTimeout','ReadTimeout','Timeout'}
DEPENDENCIES=('workspace/round17_f1_complete.py','tests/test_round17_f1_complete.py')


class RecoveryMismatch(RuntimeError):
    pass


def fingerprint():
    return dict(round16=blocks.fingerprint(),new=c.fingerprint(DEPENDENCIES))


def _events(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]


def _journal_parts(path):
    completed,pending=parent.replay_records(path)
    requests=[e for e in _events(path) if e.get('event')=='request']
    first=requests[0] if requests else None
    uncertain=[e for e in _events(path) if e.get('event')=='uncertain']
    return completed,pending,first,uncertain


def compose_history(records):
    """Return successful request/response pairs plus cumulative uncertain attempts.

    records is [(result_row, journal_path), ...] in chronological order.
    Each later attempt must start with the previous uncertain payload.
    """
    completed=[]
    extra_uncertain=0
    prior_pending=None
    for row,journal in records:
        got,pending,first,uncertain=_journal_parts(journal)
        if prior_pending is not None:
            if first is None or first.get('data')!=prior_pending.get('data'):
                raise RecoveryMismatch('Recovery attempt did not retry the prior uncertain payload')
        completed.extend(got)
        extra_uncertain+=len(uncertain)
        prior_pending=pending
        if row.get('status')!='FAILED':
            raise RecoveryMismatch('Historical recovery record must be FAILED')
    return completed,prior_pending,extra_uncertain


def same_prefix(actions,expected_actions,chains,expected_chains):
    if len(actions)!=len(expected_actions):
        return False
    for a,b in zip(actions,expected_actions):
        if a['action']!=b['action'] or list(a['args'])!=list(b['args']) or not c.receipt_equal(a['result'],b['result']):
            return False
    return parent.same_chains(chains,expected_chains)


class CumulativeReplayLLM(JournalLLM):
    def __init__(self,path,completed,pending,extra_uncertain,live=True):
        ZhipuLLM.__init__(self,key() if live else 'offline-unused-placeholder',5,model_config=config(MODEL))
        self.path=Path(path);self.stop_path=None
        self.completed=list(completed);self.pending=pending
        self.extra_uncertain=int(extra_uncertain);self.live=live
        self.cursor=0;self.boundary=None;self.counter=None;self.boundary_checked=False
        self.new_requests=0

    def _post_once(self,url,data):
        if self.cursor<len(self.completed):
            request,response=self.completed[self.cursor]
            if request['data']!=data:
                raise RecoveryMismatch('Historical successful request input changed at '+str(self.cursor+1))
            self.cursor+=1;self.http_request_count+=1
            return types.SimpleNamespace(status_code=200,text=response['body'],json=lambda:json.loads(response['body']))
        if not self.boundary_checked:
            if self.pending is not None and self.pending['data']!=data:
                raise RecoveryMismatch('Latest uncertain request input changed')
            if self.boundary is not None:self.boundary()
            self.boundary_checked=True
            for _ in range(self.extra_uncertain):
                self.http_request_count+=1
                if self.counter is not None:self.counter._reserve_call()
            if not self.live:raise parent.ReplayComplete()
        if not self.live:raise parent.ReplayComplete()
        self.new_requests+=1
        return JournalLLM._post_once(self,url,data)


def prepare():
    p=blocks.check();c.preserved()
    failed=blocks.OUT/'runs/r13-glm-036-S44.json'
    journal=failed.with_suffix('.http.jsonl')
    row=c.read(failed)
    if row.get('status')!='FAILED' or row.get('error_type')!='ServiceAbort':
        raise RecoveryMismatch('Expected round16 failed game not found')
    completed,pending=parent.replay_records(journal)
    if len(completed)!=19 or pending is None:
        raise RecoveryMismatch('Unexpected round16 SSL evidence shape')
    protected=[blocks.OUT/'protocol.json',blocks.OUT/'STOP.json',failed,journal]
    protocol=dict(code=fingerprint(),model=MODEL,model_config=config(MODEL),
        original_protocol_sha256=c.sha(blocks.OUT/'protocol.json'),
        original_failure=dict(path=failed.relative_to(c.ROOT).as_posix(),sha256=c.sha(failed),
            journal=journal.relative_to(c.ROOT).as_posix(),journal_sha256=c.sha(journal),
            actions=len(row['actions']),chains=len(row['chains']),successful_responses=len(completed),
            uncertain_requests=1),
        seeds=p['seeds'],planned_graphs=48,max_transport_recoveries=MAX_TRANSPORT_RECOVERIES,
        recovery_rule='Only transport-uncertain terminal attempts may be retried. Charge every uncertain physical attempt; exact public actions/receipts and completed chains must replay before a new request.',
        hard_stops='HTTP/credit/nontransport/payload/identity mismatch, retry cap, or resource limit.',
        no_seed_change=True,no_model_change=True,no_score_selection=True,no_upload=True)
    c.save(OUT/'protocol.json',protocol,True)
    c.save(OUT/'protected-round16.json',{str(x.relative_to(c.ROOT)).replace('\\','/'):c.sha(x) for x in protected},True)
    print(json.dumps(dict(prepared=True,model=MODEL,original_failure=protocol['original_failure'],originals=c.preserved()),ensure_ascii=False))


def check():
    p=c.read(OUT/'protocol.json')
    if p['code']!=fingerprint():raise RecoveryMismatch('Round17 F1 source drift')
    for name,h in c.read(OUT/'protected-round16.json').items():
        if c.sha(c.ROOT/name)!=h:raise RecoveryMismatch('Protected round16 evidence changed: '+name)
    return p


def existing_pass(entry,variant):
    name=entry['spec']['case']+'-'+variant+'.json';found=[]
    for folder,is_parent in ((parent.OUT/'runs',True),(blocks.OUT/'runs',False),(OUT/'final',False)):
        path=folder/name
        if not path.exists():continue
        row=c.read(path)
        if row.get('status')=='PASS':
            # Parent and blocks have different harness fingerprints but identical frozen packages.
            if row['spec']!=entry['spec'] or row['variant']!=variant or row['identity']['model_config']!=config(MODEL):
                raise RecoveryMismatch('Existing PASS identity mismatch')
            c.audit_game(row);found.append((path,row))
    if len(found)>1:raise RecoveryMismatch('Duplicate completed game')
    return found[0] if found else None


def history_for(entry,variant):
    name=entry['spec']['case']+'-'+variant
    records=[]
    old=blocks.OUT/'runs'/(name+'.json')
    if old.exists():
        row=c.read(old)
        if row.get('status')=='FAILED':
            records.append((row,old.with_suffix('.http.jsonl')))
    for i in range(MAX_TRANSPORT_RECOVERIES):
        rp=OUT/'attempts'/(name+f'-attempt{i+1}.json')
        if not rp.exists():break
        row=c.read(rp)
        if row.get('status')=='FAILED':
            records.append((row,rp.with_suffix('.http.jsonl')))
        else:
            raise RecoveryMismatch('Unexpected nonfailed attempt record')
    return records


def _latest_transport_uncertain(journal):
    ev=_events(journal)
    uncertain=[e for e in ev if e.get('event')=='uncertain']
    if not uncertain:return False
    return uncertain[-1].get('error_type') in TRANSPORT_ERRORS and uncertain[-1].get('number')==max(e['number'] for e in ev if e.get('event')=='request')


def run_attempt(entry,variant,attempt_index):
    spec=entry['spec'];name=spec['case']+'-'+variant
    records=history_for(entry,variant)
    completed,pending,extra=compose_history(records) if records else ([],None,0)
    latest_row=records[-1][0] if records else None
    target=OUT/'attempts'/(name+f'-attempt{attempt_index}.json')
    journal=target.with_suffix('.http.jsonl')
    if target.exists() or journal.exists():raise RecoveryMismatch('Attempt path already exists')
    module=parent.load(variant);module.coefficients.cache_clear()
    raw=FiniteEnv(c.generate(spec));logged=LoggedEnv(raw);chains=[]
    llm=CumulativeReplayLLM(journal,completed,pending,extra,True)
    row=dict(status='STARTED',identity=dict(entry=entry,variant=variant,model_config=config(MODEL),code=check()['code']),
             spec=spec,variant=variant,attempt=attempt_index)
    c.save(target,row)
    started=time.monotonic()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model=c.old.initialize(module,logged,mode='integration',llm=llm);llm.counter=model.llm_counter
            def boundary():
                if latest_row is None:return
                if not same_prefix(logged.actions,latest_row['actions'],chains,latest_row['chains']):
                    raise RecoveryMismatch('Public prefix or completed chains changed before retry')
            llm.boundary=boundary
            for role in ('scout','commander'):
                agent=getattr(model,role);orig=agent._run_chain
                def trace(name_,data,orig=orig,role=role):
                    before=copy.deepcopy(data);answer=orig(name_,data)
                    chains.append(dict(role=role,chain=name_,operation=model.operation_count,input=before,output=copy.deepcopy(answer)))
                    return answer
                agent._run_chain=trace
            result=c.finish(raw,logged,model)
        if records and not llm.boundary_checked:raise RecoveryMismatch('Recovery boundary not checked')
        if len(raw._scanned)!=44 or llm.http_request_count>model.llm_counter.call_count or model.llm_counter.call_count>=120:
            raise RecoveryMismatch('Recovered resource accounting mismatch')
        row.update(result,status='PASS',chains=chains,http_calls=llm.http_request_count,new_http_calls=llm.new_requests,
                   replayed_responses=llm.cursor,charged_uncertain_attempts=extra,metadata=llm.response_metadata,
                   elapsed_seconds=time.monotonic()-started)
        c.save(target,row)
        final=OUT/'final'/(name+'.json')
        c.save(final,row,True)
        return row
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,actions=logged.actions,chains=chains,
                   new_http_calls=llm.new_requests,replayed_responses=llm.cursor,charged_uncertain_attempts=extra,
                   elapsed_seconds=time.monotonic()-started)
        c.save(target,row)
        raise


def complete_one(index,variant):
    p=check();entry=p['seeds'][index]
    found=existing_pass(entry,variant)
    if found:return found[1]
    records=history_for(entry,variant)
    if records:
        last_journal=records[-1][1]
        if not _latest_transport_uncertain(last_journal):
            raise ServiceAbort('Prior failure is not recoverable transport uncertainty')
        if len(records)>MAX_TRANSPORT_RECOVERIES:
            raise ServiceAbort('Transport recovery cap exceeded')
    start_attempt=max(1,len(records))
    for attempt in range(start_attempt,MAX_TRANSPORT_RECOVERIES+1):
        try:
            return run_attempt(entry,variant,attempt)
        except ServiceAbort:
            rp=OUT/'attempts'/(entry['spec']['case']+'-'+variant+f'-attempt{attempt}.json')
            jp=rp.with_suffix('.http.jsonl')
            if not jp.exists() or not _latest_transport_uncertain(jp) or attempt>=MAX_TRANSPORT_RECOVERIES:
                c.save(OUT/'BLOCKED.json',dict(case=entry['spec']['case'],variant=variant,attempt=attempt,
                    reason='service_or_retry_cap',error_type=c.read(rp).get('error_type') if rp.exists() else 'unknown'))
                raise
            continue
    raise ServiceAbort('Recovery cap exhausted')


def run(start=36,stop=48):
    check()
    if not 0<=start<stop<=48:raise ValueError('Invalid range')
    with ExclusiveBatch(OUT/'execution.lock'):
        for index in range(start,stop):
            order=('S44','F1') if index%2==0 else ('F1','S44')
            for variant in order:
                row=complete_one(index,variant)
                print(json.dumps(dict(index=index,variant=variant,status=row['status'],
                    new_http_calls=row.get('new_http_calls',0),charged_uncertain=row.get('charged_uncertain_attempts',0))),flush=True)
    return summarize(require_all=False)


def summarize(require_all=True):
    p=check();pairs=[];completed=0;missing=[];max_error=0.
    for entry in p['seeds']:
        group={}
        for variant in ('S44','F1'):
            found=existing_pass(entry,variant)
            if found is None:
                missing.append(entry['spec']['case']+'-'+variant);continue
            row=found[1];max_error=max(max_error,c.audit_game(row));completed+=1;group[variant]=row
        if len(group)==2:
            pairs.append(dict(case=entry['spec']['case'],family=entry['spec']['family'],
                              delta=group['F1']['final_score']-group['S44']['final_score']))
    result=dict(status='COMPLETE' if not missing else 'INCOMPLETE',planned_graphs=48,completed_games=completed,
                complete_pairs=len(pairs),missing=missing,max_forward_error=max_error,model=MODEL,originals=c.preserved())
    if not missing:
        result['statistics']=c.paired_stats(pairs);result['pairs']=pairs
    c.save(OUT/'summary.json',result)
    if require_all and missing:raise RecoveryMismatch('F1 cohort incomplete')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','missing')},ensure_ascii=False),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary'])
    ap.add_argument('--start',type=int,default=36);ap.add_argument('--stop',type=int,default=48);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run(a.start,a.stop)
    else:summarize(True)

if __name__=='__main__':main()
