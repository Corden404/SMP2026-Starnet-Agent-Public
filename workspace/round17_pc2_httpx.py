"""Real-LLM validation of PC2 vs S44 using the frozen 48-graph cohort and verified httpx transport."""
from __future__ import annotations
import argparse, contextlib, copy, io, json, time, hashlib
import httpx

import round16_common as c
import round16_experiment as exp
import round17_pc2_real as old
import round17_f1_httpx as f1
import round17_replay as replay
from round11_api import ServiceAbort, config
from round9_api import append
from round9_lock import ExclusiveBatch
from zhipu import ensure_credit_available
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv

OUT=c.ROOT/'agent_notes/experiments/round17-pc2-httpx'
MODEL=f1.MODEL
MAX_TRANSPORT_ATTEMPTS=3
ALLOWED=set(f1.ALLOWED)
DEPENDENCIES=('workspace/round17_pc2_httpx.py','tests/test_round17_pc2_httpx.py',
              'workspace/round17_f1_httpx.py','workspace/round17_pc2_real.py',
              'workspace/round17_replay.py')


def fingerprint():
    return dict(core=c.fingerprint(DEPENDENCIES),
        packages={v:{n:hashlib.sha256(b).hexdigest() for n,b in exp.source_files(v).items()} for v in ('S44','PC2')})


def prepare():
    c.preserved()
    prior=c.read(old.OUT/'protocol.json')
    if list(prior['variants'])!=['S44','PC2'] or prior['planned_graphs']!=48:
        raise RuntimeError('Prior frozen PC2 real cohort changed')
    if any((old.OUT/x).exists() and any((old.OUT/x).glob('*')) for x in ('attempts','final')):
        raise RuntimeError('Superseded PC2 transport already has live results')
    if c.read(c.OUT/'resume/completion.json')['pc2_decision']['eligible_for_real'] is not True:
        raise RuntimeError('PC2 real gate not eligible')
    protocol=dict(code=fingerprint(),model=MODEL,model_config=config(MODEL),seeds=prior['seeds'],
        planned_graphs=48,variants=['S44','PC2'],max_transport_attempts=MAX_TRANSPORT_ATTEMPTS,
        supersedes_protocol_sha256=c.sha(old.OUT/'protocol.json'),
        reason='Transport-only replacement before any PC2 live game; reuses exact frozen 48 seeds/packages.',
        design=prior['design'],estimand=prior['estimand'],
        tls='Same verified httpx explicit certifi SSLContext as completed F1 validation',
        no_seed_change=True,no_model_change=True,no_policy_change=True,no_tuning=True,no_upload=True)
    c.save(OUT/'protocol.json',protocol,True)
    for v in ('S44','PC2'):exp.load(v)
    print(json.dumps(dict(prepared=True,graphs=48,model=MODEL,old_live_games=0,originals=c.preserved()),ensure_ascii=False))


def check():
    p=c.read(OUT/'protocol.json')
    if p['code']!=fingerprint():raise RuntimeError('PC2 HTTPX source drift')
    if p['supersedes_protocol_sha256']!=c.sha(old.OUT/'protocol.json'):raise RuntimeError('Superseded protocol changed')
    return p


def history_for(entry,variant):
    records=[];name=entry['spec']['case']+'-'+variant
    for i in range(1,MAX_TRANSPORT_ATTEMPTS+1):
        rp=OUT/'attempts'/(name+f'-attempt{i}.json')
        if not rp.exists():break
        row=c.read(rp)
        if row.get('status')!='FAILED':raise RuntimeError('Unexpected PC2 HTTPX attempt state')
        records.append((row,rp.with_suffix('.http.jsonl')))
    return records


def final_pass(entry,variant):
    path=OUT/'final'/(entry['spec']['case']+'-'+variant+'.json')
    if not path.exists():return None
    row=c.read(path)
    if row.get('status')!='PASS' or row['identity']['entry']!=entry or row['variant']!=variant or row['identity']['model_config']!=config(MODEL):
        raise RuntimeError('PC2 HTTPX final identity mismatch')
    c.audit_game(row);return row


def run_attempt(entry,variant,attempt):
    spec=entry['spec'];name=spec['case']+'-'+variant;records=history_for(entry,variant)
    completed,pending,extra=replay.compose_history(records,MODEL) if records else ([],None,0)
    latest=records[-1][0] if records else None
    target=OUT/'attempts'/(name+f'-attempt{attempt}.json');journal=target.with_suffix('.http.jsonl')
    if target.exists() or journal.exists():raise RuntimeError('PC2 HTTPX attempt exists')
    module=exp.load(variant);module.coefficients.cache_clear()
    if hasattr(module,'pc2_coefficients'):module.pc2_coefficients.cache_clear()
    raw=FiniteEnv(c.generate(spec));logged=LoggedEnv(raw);chains=[]
    llm=f1.HTTPXReplayLLM(journal,completed,pending,extra,True)
    row=dict(status='STARTED',identity=dict(entry=entry,variant=variant,model_config=config(MODEL),code=check()['code']),
             spec=spec,variant=variant,attempt=attempt);c.save(target,row);started=time.monotonic()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model=c.old.initialize(module,logged,mode='integration',llm=llm);llm.counter=model.llm_counter
            def boundary():
                if latest is None:return
                if not f1.base.same_prefix(logged.actions,latest['actions'],chains,latest['chains']):
                    raise RuntimeError('PC2 HTTPX public prefix/chains changed')
            llm.boundary=boundary
            for role in ('scout','commander'):
                agent=getattr(model,role);orig=agent._run_chain
                def trace(name_,data,orig=orig,role=role):
                    before=copy.deepcopy(data);ans=orig(name_,data)
                    chains.append(dict(role=role,chain=name_,operation=model.operation_count,input=before,output=copy.deepcopy(ans)));return ans
                agent._run_chain=trace
            result=c.finish(raw,logged,model)
        if records and not llm.boundary_checked:raise RuntimeError('PC2 HTTPX boundary not checked')
        if len(raw._scanned)!=44 or llm.http_request_count>model.llm_counter.call_count or model.llm_counter.call_count>=120:
            raise RuntimeError('PC2 HTTPX resource accounting mismatch')
        row.update(result,status='PASS',chains=chains,http_calls=llm.http_request_count,new_http_calls=llm.new_requests,
            replayed_responses=llm.cursor,charged_uncertain_attempts=extra,metadata=llm.response_metadata,elapsed_seconds=time.monotonic()-started)
        c.save(target,row);c.save(OUT/'final'/(name+'.json'),row,True);return row
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,actions=logged.actions,chains=chains,
            new_http_calls=llm.new_requests,replayed_responses=llm.cursor,charged_uncertain_attempts=extra,elapsed_seconds=time.monotonic()-started)
        c.save(target,row);raise
    finally:llm.close()


def complete_one(index,variant):
    p=check();entry=p['seeds'][index];done=final_pass(entry,variant)
    if done:return done
    records=history_for(entry,variant)
    if records and not replay.latest_transport_uncertain(records[-1][1],ALLOWED):
        raise ServiceAbort('Prior PC2 failure is not transport-recoverable')
    own=len(records)
    for attempt in range(own+1,MAX_TRANSPORT_ATTEMPTS+1):
        try:return run_attempt(entry,variant,attempt)
        except ServiceAbort:
            rp=OUT/'attempts'/(entry['spec']['case']+'-'+variant+f'-attempt{attempt}.json');jp=rp.with_suffix('.http.jsonl')
            if not jp.exists() or not replay.latest_transport_uncertain(jp,ALLOWED) or attempt>=MAX_TRANSPORT_ATTEMPTS:
                c.save(OUT/'BLOCKED.json',dict(case=entry['spec']['case'],variant=variant,attempt=attempt,reason='service_or_httpx_cap'));raise
    raise ServiceAbort('PC2 HTTPX retry cap exhausted')


def run(start=0,stop=48):
    p=check()
    fs=c.read(f1.OUT/'summary.json')
    if fs.get('status')!='COMPLETE' or fs.get('model')!=MODEL:raise RuntimeError('Complete F1 same-model validation required first')
    if not 0<=start<stop<=48:raise ValueError('Invalid range')
    with ExclusiveBatch(OUT/'execution.lock'):
        for index in range(start,stop):
            for variant in (('S44','PC2') if index%2==0 else ('PC2','S44')):
                row=complete_one(index,variant)
                print(json.dumps(dict(index=index,variant=variant,status=row['status'],new_http_calls=row.get('new_http_calls',0),charged_uncertain=row.get('charged_uncertain_attempts',0))),flush=True)
    return summarize(False)


def summarize(require_all=True):
    p=check();pairs=[];missing=[];rows=[];max_error=0.
    for entry in p['seeds']:
        group={}
        for variant in ('S44','PC2'):
            row=final_pass(entry,variant)
            if row is None:missing.append(entry['spec']['case']+'-'+variant);continue
            rows.append(row);group[variant]=row;max_error=max(max_error,c.audit_game(row))
        if len(group)==2:pairs.append(dict(case=entry['spec']['case'],family=entry['spec']['family'],delta=group['PC2']['final_score']-group['S44']['final_score']))
    result=dict(status='COMPLETE' if not missing else 'INCOMPLETE',planned_graphs=48,completed_games=len(rows),complete_pairs=len(pairs),
        missing=missing,max_forward_error=max_error,model=MODEL,transport='httpx-explicit-context',originals=c.preserved())
    if not missing:
        st=c.paired_stats(pairs);result.update(statistics=st,pairs=pairs,
            verdict='REAL_POSITIVE_EXPLORATORY' if st['mean']>0 else 'KEEP_S44_REAL_NONPOSITIVE',
            confirmed_real=st['ci95_t'][0]>0 and st['ci95_family_bootstrap'][0]>0)
    c.save(OUT/'summary.json',result)
    if require_all and missing:raise RuntimeError('PC2 HTTPX cohort incomplete')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','missing')},ensure_ascii=False));return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=48);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run(a.start,a.stop)
    else:summarize(True)
if __name__=='__main__':main()
