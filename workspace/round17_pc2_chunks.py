"""Score-blind chunk continuation for frozen round17 PC2 real validation.

Registered while the original 0..48 process is still running and before reading
any PC2/S44 real score deltas. It never changes seeds, packages, model, or order.
Completed parent games are reused. If the parent is externally timed out, one
STARTED game may be recovered by replaying every completed LLM response. A
terminal orphan request is conservatively charged as uncertain before retry.
"""
from __future__ import annotations
import argparse, contextlib, copy, io, json, time, hashlib
from pathlib import Path

import round16_common as c
import round16_experiment as exp
import round17_pc2_httpx as base
import round17_f1_httpx as f1
import round17_replay as replay
from round11_api import ServiceAbort, config
from round9_lock import ExclusiveBatch
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv

OUT=c.ROOT/'agent_notes/experiments/round17-pc2-chunks'
MODEL=base.MODEL
MAX_ATTEMPTS=3
DEPENDENCIES=('workspace/round17_pc2_chunks.py','tests/test_round17_pc2_chunks.py',
              'workspace/round17_pc2_httpx.py','workspace/round17_replay.py')


def fingerprint():
    return dict(core=c.fingerprint(DEPENDENCIES),
        packages={v:{n:hashlib.sha256(b).hexdigest() for n,b in exp.source_files(v).items()} for v in ('S44','PC2')})


def prepare():
    p=base.check();c.preserved()
    protocol=dict(code=fingerprint(),parent_protocol_sha256=c.sha(base.OUT/'protocol.json'),
        model=MODEL,model_config=config(MODEL),seeds=p['seeds'],planned_graphs=48,
        variants=['S44','PC2'],registered_score_blind=True,max_graphs_per_process=8,
        reason='Operational chunking registered before any real PC2/S44 delta summary was read; prevents one 60-minute harness timeout from truncating the fixed cohort.',
        timeout_rule='Reuse PASS finals. For one externally-killed STARTED attempt, replay all completed response payloads. If its last request has no response/uncertain marker, conservatively count it as one uncertain physical/logical attempt and retry the exact payload.',
        hard_stops='HTTP/credit/nontransport/payload mismatch, noncontiguous journals, >1 STARTED incomplete game, or recovery cap.',
        no_seed_change=True,no_model_change=True,no_policy_change=True,no_tuning=True,no_sample_extension=True,no_upload=True)
    c.save(OUT/'protocol.json',protocol,True)
    print(json.dumps(dict(prepared=True,score_blind=True,max_graphs_per_process=8,originals=c.preserved()),ensure_ascii=False))


def check():
    p=c.read(OUT/'protocol.json')
    if p['code']!=fingerprint():raise RuntimeError('Chunk source drift')
    if p['parent_protocol_sha256']!=c.sha(base.OUT/'protocol.json'):raise RuntimeError('Parent protocol changed')
    return p


def parent_pass(entry,variant):
    return base.final_pass(entry,variant)


def own_pass(entry,variant):
    path=OUT/'final'/(entry['spec']['case']+'-'+variant+'.json')
    if not path.exists():return None
    row=c.read(path)
    if row.get('status')!='PASS' or row['identity']['entry']!=entry or row['variant']!=variant or row['identity']['model_config']!=config(MODEL):
        raise RuntimeError('Chunk final identity mismatch')
    c.audit_game(row);return row


def existing_pass(entry,variant):
    a=parent_pass(entry,variant);b=own_pass(entry,variant)
    if a is not None and b is not None:raise RuntimeError('Duplicate PASS across parent/chunk')
    return a or b


def parse_started(journal):
    rows=replay.events(journal);req=[x for x in rows if x.get('event')=='request']
    if not req:raise RuntimeError('STARTED journal has no request')
    nums=[int(x['number']) for x in req]
    if nums!=list(range(nums[0],nums[0]+len(nums))):raise RuntimeError('STARTED request numbering gap')
    responses={int(x['number']):x for x in rows if x.get('event')=='response'}
    uncertain=[x for x in rows if x.get('event')=='uncertain']
    if len(uncertain)>1:raise RuntimeError('Multiple uncertain events in STARTED journal')
    completed=[];pending=None;extra=0
    for request in req:
        num=int(request['number']);response=responses.get(num)
        if response is not None:
            if response.get('status')!=200:raise RuntimeError('STARTED has non-200 response')
            body=json.loads(response.get('body','{}'))
            if body.get('model')!=MODEL:raise RuntimeError('STARTED model mismatch')
            completed.append((request,response))
        elif uncertain and int(uncertain[0]['number'])==num and num==nums[-1]:
            pending=request;extra=1
        elif num==nums[-1]:
            # Process timeout may kill the request before the transport exception
            # handler can append an uncertain record. Charge it conservatively.
            pending=request;extra=1
        else:raise RuntimeError('Nonterminal STARTED request lacks response')
    return completed,pending,extra


def inherited_history(entry,variant):
    name=entry['spec']['case']+'-'+variant
    attempts=sorted((base.OUT/'attempts').glob(name+'-attempt*.json'))
    failed=[];started=[]
    for rp in attempts:
        row=c.read(rp)
        if row.get('status')=='FAILED':failed.append((row,rp.with_suffix('.http.jsonl')))
        elif row.get('status')=='STARTED':started.append((row,rp.with_suffix('.http.jsonl')))
        elif row.get('status')=='PASS':pass
        else:raise RuntimeError('Unexpected parent attempt status')
    if len(started)>1:raise RuntimeError('More than one parent STARTED attempt')
    completed=[];pending=None;extra=0
    if failed:
        completed,pending,extra=replay.compose_history(failed,MODEL)
    if started:
        got,p2,e2=parse_started(started[0][1])
        first=replay.events(started[0][1])
        first_req=next(x for x in first if x.get('event')=='request')
        if pending is not None and first_req.get('data')!=pending.get('data'):
            raise RuntimeError('STARTED does not retry prior uncertain payload')
        completed.extend(got);pending=p2;extra+=e2
    return completed,pending,extra,bool(started)


def own_failed(entry,variant):
    name=entry['spec']['case']+'-'+variant;out=[]
    for i in range(1,MAX_ATTEMPTS+1):
        rp=OUT/'attempts'/(name+f'-attempt{i}.json')
        if not rp.exists():break
        row=c.read(rp)
        if row.get('status')!='FAILED':raise RuntimeError('Unexpected chunk attempt state')
        out.append((row,rp.with_suffix('.http.jsonl')))
    return out


def history(entry,variant):
    completed,pending,extra,had_started=inherited_history(entry,variant)
    own=own_failed(entry,variant)
    if own:
        got,p2,e2=replay.compose_history(own,MODEL)
        first_req=replay.journal_parts(own[0][1],MODEL)[2]
        if pending is not None and first_req.get('data')!=pending.get('data'):
            raise RuntimeError('Chunk recovery does not retry inherited pending payload')
        completed.extend(got);pending=p2;extra+=e2
    return completed,pending,extra,had_started


def run_attempt(entry,variant,attempt):
    spec=entry['spec'];name=spec['case']+'-'+variant
    completed,pending,extra,had_started=history(entry,variant)
    target=OUT/'attempts'/(name+f'-attempt{attempt}.json');journal=target.with_suffix('.http.jsonl')
    if target.exists() or journal.exists():raise RuntimeError('Chunk attempt exists')
    module=exp.load(variant);module.coefficients.cache_clear()
    if hasattr(module,'pc2_coefficients'):module.pc2_coefficients.cache_clear()
    raw=FiniteEnv(c.generate(spec));logged=LoggedEnv(raw);chains=[]
    llm=f1.HTTPXReplayLLM(journal,completed,pending,extra,True)
    row=dict(status='STARTED',identity=dict(entry=entry,variant=variant,model_config=config(MODEL),code=check()['code']),
             spec=spec,variant=variant,attempt=attempt,inherited_started=had_started)
    c.save(target,row);started=time.monotonic()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model=c.old.initialize(module,logged,mode='integration',llm=llm);llm.counter=model.llm_counter
            for role in ('scout','commander'):
                agent=getattr(model,role);orig=agent._run_chain
                def trace(name_,data,orig=orig,role=role):
                    before=copy.deepcopy(data);ans=orig(name_,data)
                    chains.append(dict(role=role,chain=name_,operation=model.operation_count,input=before,output=copy.deepcopy(ans)));return ans
                agent._run_chain=trace
            result=c.finish(raw,logged,model)
        if completed and not llm.boundary_checked:raise RuntimeError('Replay boundary was not checked')
        if len(raw._scanned)!=44 or llm.http_request_count>model.llm_counter.call_count or model.llm_counter.call_count>=120:
            raise RuntimeError('Chunk resource accounting mismatch')
        row.update(result,status='PASS',chains=chains,http_calls=llm.http_request_count,new_http_calls=llm.new_requests,
            replayed_responses=llm.cursor,charged_uncertain_attempts=extra,metadata=llm.response_metadata,elapsed_seconds=time.monotonic()-started)
        c.save(target,row);c.save(OUT/'final'/(name+'.json'),row,True);return row
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,actions=logged.actions,chains=chains,
            new_http_calls=llm.new_requests,replayed_responses=llm.cursor,charged_uncertain_attempts=extra,elapsed_seconds=time.monotonic()-started)
        c.save(target,row);raise
    finally:llm.close()


def complete_one(index,variant):
    p=check();entry=p['seeds'][index];done=existing_pass(entry,variant)
    if done:return done
    own=own_failed(entry,variant)
    if own and not replay.latest_transport_uncertain(own[-1][1],base.ALLOWED):
        raise ServiceAbort('Chunk prior failure not transport-recoverable')
    for attempt in range(len(own)+1,MAX_ATTEMPTS+1):
        try:return run_attempt(entry,variant,attempt)
        except ServiceAbort:
            rp=OUT/'attempts'/(entry['spec']['case']+'-'+variant+f'-attempt{attempt}.json');jp=rp.with_suffix('.http.jsonl')
            if not jp.exists() or not replay.latest_transport_uncertain(jp,base.ALLOWED) or attempt>=MAX_ATTEMPTS:
                c.save(OUT/'BLOCKED.json',dict(case=entry['spec']['case'],variant=variant,attempt=attempt,reason='service_or_chunk_cap'));raise
    raise ServiceAbort('Chunk recovery cap exhausted')


def run(start,stop):
    check()
    if not 0<=start<stop<=48 or stop-start>8:raise ValueError('Chunk must cover 1..8 graph indices')
    with ExclusiveBatch(OUT/'execution.lock'):
        for index in range(start,stop):
            for variant in (('S44','PC2') if index%2==0 else ('PC2','S44')):
                row=complete_one(index,variant)
                print(json.dumps(dict(index=index,variant=variant,status=row['status'],new_http_calls=row.get('new_http_calls',0),
                    charged_uncertain=row.get('charged_uncertain_attempts',0))),flush=True)
    return summarize(False)


def summarize(require_all=True):
    p=check();pairs=[];missing=[];rows=[];max_error=0.
    for entry in p['seeds']:
        group={}
        for variant in ('S44','PC2'):
            row=existing_pass(entry,variant)
            if row is None:missing.append(entry['spec']['case']+'-'+variant);continue
            rows.append(row);group[variant]=row;max_error=max(max_error,c.audit_game(row))
        if len(group)==2:pairs.append(dict(case=entry['spec']['case'],family=entry['spec']['family'],delta=group['PC2']['final_score']-group['S44']['final_score']))
    result=dict(status='COMPLETE' if not missing else 'INCOMPLETE',planned_graphs=48,completed_games=len(rows),complete_pairs=len(pairs),
        missing=missing,max_forward_error=max_error,model=MODEL,transport='httpx-explicit-context-chunked',originals=c.preserved())
    if not missing:
        st=c.paired_stats(pairs);result.update(statistics=st,pairs=pairs,
            verdict='REAL_POSITIVE_EXPLORATORY' if st['mean']>0 else 'KEEP_S44_REAL_NONPOSITIVE',
            confirmed_real=st['ci95_t'][0]>0 and st['ci95_family_bootstrap'][0]>0)
    c.save(OUT/'summary.json',result)
    if require_all and missing:raise RuntimeError('PC2 chunk cohort incomplete')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','missing')},ensure_ascii=False));return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary'])
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=8);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run(a.start,a.stop)
    else:summarize(True)

if __name__=='__main__':main()
